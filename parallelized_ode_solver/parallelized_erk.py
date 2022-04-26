from __future__ import division
from numba import cuda
import numpy as np
import math
import logging;
logging.disable(logging.WARNING)

# @cuda.jit('void(int32, int32[:], int32[:], float32[:], float32[:], float32[:], float32, float32, float32[:])')
@cuda.jit
def spmv_csr_kernel(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector, dt, c_stage, states):
    
    idx = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    if(idx<dim):
        res = 0.0
        lower, upper = csrRowPtr[idx], csrRowPtr[idx+1]
        for i in range(lower, upper):
            res += csrData[i]*inVector[csrColIdx[i]]
        res *= dt

        cuda.syncthreads()
        states[idx] += c_stage*res
        outVector[idx] = res

# @cuda.jit('void(int32, int32[:], int32[:], int32[:], int32[:], float32[:], float32[:], float32[:], float32, float32, float32[:])')
@cuda.jit
def spmv_jds_kernel(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, inVector, outVector, dt, c_stage, states):
    
    # idx = cuda.grid(1)
    idx = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    if(idx<dim):
        res = 0.0
        for i in range(jdsRowNNZ[idx]):
            jdx = idx + jdsColStartIdx[i]
            res += jdsData[jdx]*inVector[jdsColIdx[jdx]]
        res *= dt

        cuda.syncthreads()
        states[idx] += c_stage*res
        outVector[jdsRowPerm[idx]] = res


def parallelized_erk(butcher_tableau, matrix_data, basis_fns, f_0, last_time, dt, maxThreadsPerBlock):
    c = butcher_tableau.c
    b = butcher_tableau.b
    A = butcher_tableau.A

    if matrix_data.name == 'csr':
        csrRowPtr, csrColIdx, csrData = matrix_data.csrRowPtr, matrix_data.csrColIdx, matrix_data.csrData
        csrRowPtr, csrColIdx, csrData = cuda.to_device(csrRowPtr), cuda.to_device(csrColIdx), cuda.to_device(csrData)

    elif matrix_data.name == 'jds':
        jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = matrix_data.jdsRowPerm, matrix_data.jdsRowNNZ, matrix_data.jdsColStartIdx, matrix_data.jdsColIdx, matrix_data.jdsData
        jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = cuda.to_device(jdsRowPerm), cuda.to_device(jdsRowNNZ), cuda.to_device(jdsColStartIdx), cuda.to_device(jdsColIdx), cuda.to_device(jdsData)

    order = len(b)
    
    Nstep = int(last_time/dt)
    states_dim = len(f_0)
    basis_fi = basis_fns(0, f_0)
    basis_dim = len(basis_fi)

    t0 = 0.
    for _ in range(Nstep):
        states = np.zeros(states_dim, dtype = 'float32')
        for i in range(states_dim):
            states[i] = f_0[i]

        ks = []
        for stage in range(order):
            ti = t0 + b[stage]*dt
            f_tmp = np.zeros(states_dim)
            for i in range(states_dim):
                fi = f_0[i]
                for a_idx in range(len(ks)):
                    fi += A[stage][a_idx]*ks[a_idx][i]
                f_tmp[i] = fi
            
            basis_fi = basis_fns(ti, f_tmp)

            ki = np.zeros(states_dim, dtype = 'float32')
            blockspergrid = math.ceil(states_dim / maxThreadsPerBlock)

            basis_fi = cuda.to_device(basis_fi)
            d_states = cuda.to_device(states)
            d_ki = cuda.to_device(ki)
            if matrix_data.name == 'csr':
              spmv_csr_kernel[blockspergrid, maxThreadsPerBlock](states_dim, csrRowPtr, csrColIdx, csrData, basis_fi, d_ki, dt, c[stage], d_states)
            elif matrix_data.name == 'jds':
              spmv_jds_kernel[blockspergrid, maxThreadsPerBlock](states_dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, basis_fi, d_ki, dt, c[stage], d_states)

            ki = d_ki.copy_to_host()
            states = d_states.copy_to_host()

            ks.append(ki)

        t0 += dt
        f_0 = states

    return f_0


if __name__ == "__main__":
    from time import time
    from data_generator import *
    from data_structure import *

    c = [1/6, 1/3, 1/3, 1/6]
    b = [0, 1/2, 1/2, 1]
    A = [[0  ,   0, 0, 0],
         [1/2,   0, 0, 0],
         [0  , 1/2, 0, 0],
         [0  ,   0, 1, 0]]
    rk4 = butcher_tableau(c, b, A)
    gpu = cuda.get_current_device()
    maxThreadsPerBlock = gpu.MAX_THREADS_PER_BLOCK

    def linear_basis(t, vector):
        return vector
    dim, csrRowPtr, csrColIdx, csrData = generateCSRMatrix(15)
    csrRowPtr, csrColIdx, csrData = np.array(csrRowPtr), np.array(csrColIdx), np.array(csrData)

    dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = csr2jds(dim, csrRowPtr, csrColIdx, csrData)
    jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = np.array(jdsRowPerm), np.array(jdsRowNNZ), np.array(jdsColStartIdx), np.array(jdsColIdx), np.array(jdsData)


    f_0 = np.array([1. for _ in range(dim)])
    csr_matrix_data = csr(dim, csrRowPtr, csrColIdx, csrData)
    st = time()
    _ = parallelized_erk(rk4, csr_matrix_data, linear_basis, f_0, 1, 1, maxThreadsPerBlock)
    ed = time()
    print("CSR Time elapse: {}".format(ed-st))

    f_0 = np.array([1. for _ in range(dim)])
    jds_matrix_data = jds(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData)
    st = time()
    _ = parallelized_erk(rk4, jds_matrix_data, linear_basis, f_0, 1, 1, maxThreadsPerBlock)
    ed = time()
    print("JDS Time elapse: {}".format(ed-st))

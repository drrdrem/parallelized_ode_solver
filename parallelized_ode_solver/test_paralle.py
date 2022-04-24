from data_generator import *
from data_structure import *
from serial_erk import *
from parallelized_erk import *
from numba import cuda

gpu = cuda.get_current_device()
maxThreadsPerBlock = gpu.MAX_THREADS_PER_BLOCK

dim = 15
blockspergrid = math.ceil(dim / maxThreadsPerBlock)
dt = 0.1
weight = 0.5


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


dim, csrRowPtr, csrColIdx, csrData = generateCSRMatrix(dim)
csrRowPtr, csrColIdx, csrData = np.array(csrRowPtr), np.array(csrColIdx), np.array(csrData)


dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = csr2jds(dim, csrRowPtr, csrColIdx, csrData)
jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = np.array(jdsRowPerm), np.array(jdsRowNNZ), np.array(jdsColStartIdx), np.array(jdsColIdx), np.array(jdsData)

csrRowPtr, csrColIdx, csrData = cuda.to_device(csrRowPtr), cuda.to_device(csrColIdx), cuda.to_device(csrData)
jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = cuda.to_device(jdsRowPerm), cuda.to_device(jdsRowNNZ), cuda.to_device(jdsColStartIdx), cuda.to_device(jdsColIdx), cuda.to_device(jdsData)

ki = np.zeros(dim, dtype = 'float32')
states = np.zeros(dim, dtype = 'float32')
basis_fi = np.array([1. for _ in range(dim)])
basis_fi = cuda.to_device(basis_fi)
d_states = cuda.to_device(states)
d_ki = cuda.to_device(ki)
spmv_csr_kernel[blockspergrid, maxThreadsPerBlock](dim, csrRowPtr, csrColIdx, csrData, basis_fi, d_ki, dt, weight, d_states)
ki = d_ki.copy_to_host()
states = d_states.copy_to_host()
print('csr:')
print(states)
print(ki)
ki = np.zeros(dim, dtype = 'float32')
states = np.zeros(dim, dtype = 'float32')
basis_fi = np.array([1. for _ in range(dim)])
basis_fi = cuda.to_device(basis_fi)
d_states = cuda.to_device(states)
d_ki = cuda.to_device(ki)
spmv_jds_kernel[blockspergrid, maxThreadsPerBlock](dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, basis_fi, d_ki, dt, weight, d_states)
ki = d_ki.copy_to_host()
states = d_states.copy_to_host()
print('jds:')
print(states)
print(ki)
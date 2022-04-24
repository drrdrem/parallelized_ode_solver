from data_generator import *
from data_structure import *
from serial_erk import *
from parallelized_erk import *
from numba import cuda

import matplotlib.pyplot as plt

from time import time


c = [1/6, 1/3, 1/3, 1/6]
b = [0, 1/2, 1/2, 1]
A = [[0  ,   0, 0, 0],
        [1/2,   0, 0, 0],
        [0  , 1/2, 0, 0],
        [0  ,   0, 1, 0]]
rk4 = butcher_tableau(c, b, A)

def linear_basis(t, states):
    return states

gpu = cuda.get_current_device()
maxThreadsPerBlock = gpu.MAX_THREADS_PER_BLOCK

step = 1000

serial_elapsed_all, csr_elapsed_all, jds_elapsed_all = [], [], []
for dim in range(10, 10010, step):
    serial_elapsed, csr_elapsed, jds_elapsed = [], [], []
    for _ in range(30):
        dim, csrRowPtr, csrColIdx, csrData = generateCSRMatrix(dim)
        f_0 = [0. for _ in range(dim)]

        st = time()
        _ = erk(rk4, csrRowPtr, csrColIdx, csrData, linear_basis, f_0, 1, 0.1)
        ed = time()

        serial_elapsed.append(ed - st)

        csrRowPtr, csrColIdx, csrData = np.array(csrRowPtr), np.array(csrColIdx), np.array(csrData)
        dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = csr2jds(dim, csrRowPtr, csrColIdx, csrData)
        jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = np.array(jdsRowPerm), np.array(jdsRowNNZ), np.array(jdsColStartIdx), np.array(jdsColIdx), np.array(jdsData)
        
        f_0 = [0. for _ in range(dim)]
        f_0 = np.array(f_0)
        csr_matrix_data = csr(dim, csrRowPtr, csrColIdx, csrData)
        st = time()
        _ = parallelized_erk(rk4, csr_matrix_data, linear_basis, f_0, 1, 0.1, maxThreadsPerBlock)
        ed = time()
        csr_elapsed.append(ed - st)

        jds_matrix_data = jds(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData)
        f_0 = [0. for _ in range(dim)]
        f_0 = np.array(f_0)
        st = time()
        _ = parallelized_erk(rk4, jds_matrix_data, linear_basis, f_0, 1, 0.1, maxThreadsPerBlock)
        ed = time()
        jds_elapsed.append(ed - st)
    serial_elapsed_all.append(serial_elapsed)
    csr_elapsed_all.append(csr_elapsed)
    jds_elapsed_all.append(jds_elapsed)

serial_elapsed_all = np.array(serial_elapsed_all).T 
csr_elapsed_all = np.array(csr_elapsed_all).T 
jds_elapsed_all = np.array(jds_elapsed_all).T 

plt.figure()
plt.errorbar(range(2, 10002, step), serial_elapsed_all.mean(axis=0), serial_elapsed_all.std(axis=0), label = 'serial csr')
plt.errorbar(range(2, 10002, step), csr_elapsed_all.mean(axis=0), csr_elapsed_all.std(axis=0), label = 'parallelized csr')
plt.errorbar(range(2, 10002, step), jds_elapsed_all.mean(axis=0), jds_elapsed_all.std(axis=0), label = 'parallelized jds')
plt.legend()
plt.xlabel("Model Compexity")
plt.ylabel("Time Elapsed (Sec)")
plt.savefig('Time_Comarison.png')
print('End of Comparison')




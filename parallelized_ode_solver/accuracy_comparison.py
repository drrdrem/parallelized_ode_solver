from data_generator import *
from data_structure import *
from serial_erk import *
from serial_parareal import *
from config import *

from tqdm import *
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

dim, csrRowPtr, csrColIdx, csrData = generateCSRMatrix(15)
coarse_alg = (rk4, 0.05) 
fine_alg = (rk4, 0.001) 
K = 50
f_0 = [1. for _ in range(dim)]

def linear_basis(t, states):
    return states

matrix = csr2matrix(dim, csrRowPtr, csrColIdx, csrData)

parareal_res = parareal(csrRowPtr, csrColIdx, csrData, linear_basis, f_0, K, 0.1, coarse_alg, fine_alg)

erk_res = erk(rk4, csrRowPtr, csrColIdx, csrData, linear_basis, f_0, 1, 0.1)


def model_fn(t, yy, dim, csrRowPtr, csrColIdx, csrData):
    f = []
    for idx in range(dim):
        lower, upper = csrRowPtr[idx], csrRowPtr[idx+1]
        res = 0.
        for i in range(lower, upper):
            res += csrData[i]*yy[csrColIdx[i]]
        f.append(res)
    return f


ground_sol = solve_ivp(model_fn, [0, 1.], np.squeeze(np.array(f_0)), args=(dim, csrRowPtr, csrColIdx, csrData), 
                        rtol=1e-3, dense_output=True)

ground_truth = ground_sol.sol([-1]).T.tolist()[0]

def mertric(res):
    val = 0
    n = len(res)
    for i in range(n):
        val += (res[i] - ground_truth[i])**2
    return val/n

dt = 1
accuracy = []
for _ in tqdm(range(10)):
    dt *= 0.1
    erk_res = erk(rk4, csrRowPtr, csrColIdx, csrData, linear_basis, f_0, 1, dt)
    
    accuracy.append(mertric(erk_res))

print(accuracy)

plt.figure()
plt.plot(accuracy)
plt.xlabel("Time Difference")
plt.ylabel("MSE")
plt.savefig('Accuracy_Comparison.png')
print('End of Comparison')


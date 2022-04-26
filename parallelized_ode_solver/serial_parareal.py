'''Serial Version of ParaReal Algorithm

'''
from serial_erk import *

def parareal(csrRowPtr, csrColIdx, csrData, basis_fns, f_0, K, dt, coarse_alg, fine_alg, last_time=1.0):

    fin_butcher_tableau, fin_dt = fine_alg[0], fine_alg[1]
    coars_butcher_tableau, coars_dt = coarse_alg[0], coarse_alg[1]

    Nstep = int((last_time)/dt)
    N_states = int(len(f_0))

    xs = [f_0]
    x_t = f_0
    for _ in range(Nstep):
        x_t = erk(coars_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, x_t, dt, coars_dt)
        xs.append(x_t)

    for k in range(K):
        xks = [f_0]
        for t in range(Nstep):
            h = erk(fin_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, xs[t], dt, fin_dt)
            j = erk(coars_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, xs[t], dt, coars_dt)
            g = erk(coars_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, xks[-1], dt, coars_dt)
            
            x_t = [g[i] + h[i] - j[i] for i in range(N_states)]
            xks.append(x_t)
        xs = xks
    return xs[-1]

if __name__ == "__main__":
    from time import time
    from data_generator import *
    c = [1/6, 1/3, 1/3, 1/6]
    b = [0, 1/2, 1/2, 1]
    A = [[0  ,   0, 0, 0],
         [1/2,   0, 0, 0],
         [0  , 1/2, 0, 0],
         [0  ,   0, 1, 0]]
    rk4 = butcher_tableau(c, b, A)

    # First example linear
    def linear_basis(t, states):
        return states

    dim, csrRowPtr, csrColIdx, csrData = generateCSRMatrix(15)

    coarse_alg = (rk4, 0.05) 
    fine_alg = (rk4, 0.001) 
    K = 50
    f_0 = [1. for _ in range(dim)]
    st = time()
    serial_res = parareal(csrRowPtr, csrColIdx, csrData, linear_basis, f_0, K, 0.1, coarse_alg, fine_alg)

    ed = time()
    print(serial_res)
    print("Serial Time elapse: {}".format(ed-st))
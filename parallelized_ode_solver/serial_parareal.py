'''
Unfinished
'''
from serial_erk import *

def parareal(csrRowPtr, csrColIdx, csrData, basis_fns, f_0, K, dt, coarse_alg, fine_alg, last_time=1.0):

    fin_butcher_tableau, fin_dt = fine_alg[0], fine_alg[1]
    coars_butcher_tableau, coars_dt = coarse_alg[0], coarse_alg[1]

    Nstep = (last_time)/dt

    xs = [f_0]
    for _ in range(Nstep):
        x_t = erk(coars_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, f_0, dt, coars_dt)
        xs.append(xs)

    xks = [f_0]
    for k in range(K):
        for t in range(Nstep):
            h = erk(fin_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, xs[t+1], dt, fin_dt)
            j = erk(coars_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, xs[t], dt, coars_dt)
            g = erk(coars_butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, xks[-1], dt, coars_dt)

            x_t = g + h - j 
            xks.append(x_t)
        xs = xks
    return xs[-1]
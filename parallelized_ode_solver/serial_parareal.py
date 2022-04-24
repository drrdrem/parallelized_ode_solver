'''
Unfinished
'''

import numpy as np
from functools import partial as fp

from serial_erk import *

def parareal(coarse_alg, fine_alg, p, K, d_t, initial_value, last_time=1.0):
    steps = int((last_time)/d_t)
    steps -= steps % p

    dt_coarse = (last_time)/p

    vec = coarse_func(t_start, t_stop, u_0, delta_t_coarse)
    last_coarse = coarse_func(t_start, t_stop, u_0, delta_t_coarse)
    for k in range(K):
        updated_coarse = np.zeros(p+1)
        tmp = np.zeros(p+1)
        tmp[0] = vec[0] # initial value
        for p, time in zip(range(p), time_intervals):
            start, stop = time
            fine_calc = fine_func(start, stop, vec[p],delta_t)
            delta_q = fine_calc[-1] - last_coarse[p+1]
            new_q = coarse_func(start, stop, tmp[p], delta_t_coarse)
            updated_coarse[p+1] = new_q[-1]
            tmp[p+1] = (updated_coarse[p+1] - last_coarse[p+1]) + fine_calc[-1]

        vec = tmp
        last_coarse = updated_coarse
    return vec
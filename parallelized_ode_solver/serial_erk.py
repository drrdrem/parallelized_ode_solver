from data_structure import *


def spmv(states_dim, csrRowPtr, csrColIdx, csrData, basis_fi, dt, c_stage, states):
    ki = []
    for idx in range(states_dim):
        lower, upper = csrRowPtr[idx], csrRowPtr[idx+1]
        kval = 0.
        for i in range(lower, upper):
            kval += csrData[i]*basis_fi[csrColIdx[i]]
        kval *= dt
        states[idx] += c_stage*kval
        ki.append(kval)
    return ki, states


def erk(butcher_tableau, csrRowPtr, csrColIdx, csrData, basis_fns, f_0, last_time, dt):
    '''Explict Runge Kutta
    Args
        butcher_tableau: butcher_tableau for different order/ stage of explicit runge kutta.
        weight_matrix (2D list): weight_matrix for f(t, x). dimension is (# of states, # of feature).
        basis_fns: feature function basis for f(t, x). dimension is (# of feature).
        f_0 (list): initial values. dimension is (# of states).
        last_time (float): final time for numerical evaluation.
        dt (float): delt t.
    '''
    c = butcher_tableau.c
    b = butcher_tableau.b
    A = butcher_tableau.A

    order = len(b)
    
    Nstep = int(last_time//dt)
    states_dim = len(f_0)
    basis_fi = basis_fns(0, f_0)
    basis_dim = len(basis_fi)

    t0 = 0.
    for _ in range(Nstep):
        states = []
        for i in range(states_dim):
            states.append(f_0[i])

        ks = []
        for stage in range(order):
            ti = t0 + b[stage]*dt
            f_tmp = [0 for _ in range(states_dim)]
            for i in range(states_dim):
                fi = f_0[i]
                for a_idx in range(len(ks)):
                    fi += A[stage][a_idx]*ks[a_idx][i]
                f_tmp[i] = fi
            
            basis_fi = basis_fns(ti, f_tmp)
            ki, states = spmv(states_dim, csrRowPtr, csrColIdx, csrData, basis_fi, dt, c[stage], states)
            ks.append(ki)

        t0 += dt
        f_0 = states

    return f_0


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

    f_0 = [1. for _ in range(dim)]
    st = time()
    serial_res = erk(rk4, csrRowPtr, csrColIdx, csrData, linear_basis, f_0, 1, 0.1)
    ed = time()
    print(serial_res)
    print("Serial Time elapse: {}".format(ed-st))


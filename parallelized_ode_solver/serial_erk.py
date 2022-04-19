class csr(object):
    def __init__(self):
        pass


class butcher_tableau(object):
    def __init__(self, c, b, A):
        self.c = c
        self.b = b
        self.A = A


def erk(butcher_tableau, weight_matrix, basis_fns, f_0, last_time, dt):
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
            
            ki = []
            basis_fi = basis_fns(ti, f_tmp)
            for i in range(states_dim):
                kval = 0.
                for j in range(basis_dim):
                    kval += weight_matrix[i][j]*basis_fi[j]
                kval *= dt
                ki.append(kval)
                states[i] += c[stage]*kval
            ks.append(ki)

        t0 += dt
        f_0 = states

    return f_0


if __name__ == "__main__":
    from time import time
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

    w_0 = [[-3, 0],
           [0, -2]]
    f_0 = [10, 5]
    print(erk(rk4, w_0, linear_basis, f_0, 1, 0.1))


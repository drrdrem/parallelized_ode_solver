from data_structure import *

c = [1/6, 1/3, 1/3, 1/6]
b = [0, 1/2, 1/2, 1]
A = [[0  ,   0, 0, 0],
        [1/2,   0, 0, 0],
        [0  , 1/2, 0, 0],
        [0  ,   0, 1, 0]]
rk4 = butcher_tableau(c, b, A)
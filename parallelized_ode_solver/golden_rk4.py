def rk4(y0, xn, h, w, f):
    
    # Calculating step size
    n = int(xn//h)
    t0 = 0
    c = [1/6, 1/3, 1/3, 1/6]
    b = [0, 1/2, 1/2, 1]
    A = [[0  ,   0, 0, 0],
          [1/2,   0, 0, 0],
          [0  , 1/2, 0, 0],
          [0  ,   0, 1, 0]]
    for i in range(n):
      yn = y0
      ks = []
      for j in range(4):
        ti = t0 + b[j]*h
        yi = y0
        for k in range(len(ks)):
          yi += A[j][k]*ks[k]
        # print(yi)
        ki = h*w*f(ti, yi)
        # print(ki)
        yn += c[j]*ki
        ks.append(ki)

      y0 = yn
      t0 = t0+h
    
    return yn
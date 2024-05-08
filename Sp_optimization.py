import numpy as np

def Updating(y):
    
    D = len(y)
    u = np.sort(y)[::-1]
    # print(u)
    # print("+++++++++++++++++++++++++")
    rho = 0
    for j in range(D):
        if u[j] + 1 / (j+1) * (1 - np.sum(u[0:j+1])) > 0:
            rho = j + 1

    lambda_val = 1.0 / rho * (1 - np.sum(u[0:rho]))

    x = np.zeros([D])
    for i in range(D):
        x[i] = max(y[i] + lambda_val, 0)
    
    return(x)
import numpy as np

def nor(temp_mat, temp_I):
    temp_mat  = temp_mat - np.diag(np.diag(temp_mat))
    temp_diagmat = np.diag((np.dot(temp_mat,temp_I) + 0.000000001) ** (-0.5)) 
    temp_diagmat = np.asmatrix(temp_diagmat)
    nor_A = temp_diagmat * np.asmatrix(temp_mat) * temp_diagmat
    return nor_A
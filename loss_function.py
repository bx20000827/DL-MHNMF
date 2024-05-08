import numpy as np

# def Loss_fuc(A, Uno, Uo, V_, Vo, lam, theta, View_num, n, H):
def Loss_fuc(A, Uno, Uo, V_, Vo, lam, theta, alpha, View_num, n, H):
    
    temp_loss = 0
    
    for View_order in range(View_num):
        
        temp_error = (np.linalg.norm((A[View_order] - Uno[View_order] * V_ - Uo[View_order] * Vo[View_order]), ord = 'fro'))**2
        
        temp_graph_regularization = np.trace(V_ * (np.asmatrix(np.eye(n)) - A[View_order]) * V_.T)
        
        # HSIC
        temp_diverse = 0
        temp_SumHKH = np.asmatrix(np.zeros((n,n)))
        for i in range(View_num):
            if i != View_order:
                K_w = Vo[i].T * Vo[i]
                temp_SumHKH += H * K_w * H
        temp_diverse += np.trace(Vo[View_order] * temp_SumHKH * Vo[View_order].T)
        
        temp_loss += temp_error + lam * temp_graph_regularization + theta * temp_diverse

    temp_st = 2 * alpha * sum(np.linalg.norm(V_, ord = 2, axis = 1))
    
    Loss = temp_loss + temp_st
    
    return Loss
    # return temp_loss
        
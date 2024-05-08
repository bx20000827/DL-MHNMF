import numpy as np


def Updating_Uno(A, Uno, Uo, V_, Vo, View_index):
    
    AV_ = A[View_index] * V_.T
    UoVoV_ = Uo[View_index] * Vo[View_index] * V_.T
    UnoV_V_ = Uno[View_index] * V_ * V_.T
    
    temp_mat = UoVoV_ + UnoV_V_
    temp_mat[temp_mat < np.spacing(1)] = np.spacing(1)
    step_Uno = np.divide(AV_ , temp_mat)       
    Uno[View_index] = np.multiply(Uno[View_index] , step_Uno)
    
    return Uno
    
    
    
    
def Updating_Uo(A, Uno, Uo,  Vo, V_, View_index):
    
    AVo = A[View_index] * Vo[View_index].T
    
    UoVoVo = Uo[View_index] * Vo[View_index] * Vo[View_index].T
    
    UnoV_Vo = Uno[View_index] * V_ * Vo[View_index].T
    
    temp_mat = UoVoVo + UnoV_Vo
    temp_mat[temp_mat < np.spacing(1)] = np.spacing(1)
    
    step_Uo = np.divide(AVo , temp_mat)       
    Uo[View_index] = np.multiply(Uo[View_index] , step_Uo)
    
    return Uo

    
    
    
def Updating_Vo(A, Uno, Uo, V_, Vo, theta, View_index, View_num, n, M):
    
    temp_SumKw = np.asmatrix(np.zeros((n,n)))
    for i in range(View_num):
        if i != View_index:
            K_w = Vo[i].T * Vo[i]
            temp_SumKw += K_w
    
    # 分子
    UoA = Uo[View_index].T * A[View_index]
    
    VoKwM = Vo[View_index] * temp_SumKw * M
    
    VoMKw = Vo[View_index] * M * temp_SumKw
    
    # 分母
    UoUoVo = Uo[View_index].T * Uo[View_index] * Vo[View_index]
    
    UoUnoV_ = Uo[View_index].T * Uno[View_index] * V_
    
    VoKw = Vo[View_index] * temp_SumKw
    
    VoMKwM = VoMKw * M
    
    temp_mat = UoUoVo + UoUnoV_ + theta * (VoKw + VoMKwM)
    temp_mat[temp_mat < np.spacing(1)] = np.spacing(1)
    
    step_Vo = np.divide(UoA + theta * (VoKwM + VoMKw), temp_mat)
    
    Vo[View_index] = np.multiply(Vo[View_index], step_Vo)
    
    return Vo




# def Updating_V_(A, Uno, Uo, Vo, V_, lam, View_index, View_num, k, n):
def Updating_V_(A, Uno, Uo, Vo, V_, lam, alpha, View_index, View_num, k, n):
    SumUnoA = np.asmatrix(np.zeros((k,n)))
    SumV_A = np.asmatrix(np.zeros((k,n)))
    SumUnoUoVo = np.asmatrix(np.zeros((k,n)))
    SumUnoUnoV_ = np.asmatrix(np.zeros((k,n)))
    for i in range(View_num):
        SumUnoA += Uno[i].T * A[i]
        SumV_A += V_ * A[View_index]
        SumUnoUoVo += Uno[i].T * Uo[i] * Vo[i]
        SumUnoUnoV_ += Uno[i].T * Uno[i] * V_
        
    l21Vno = alpha * np.asmatrix(np.diag((np.linalg.norm(V_ , ord=2 , axis = 1) + 0.0000000001) ** (-1))) * V_
    
    temp_mat = SumUnoUoVo + SumUnoUnoV_ + lam * View_num * V_ + alpha * l21Vno
    temp_mat[temp_mat < np.spacing(1)] = np.spacing(1)
    
    # step_V_ = np.divide(SumUnoA + lam * SumV_A, temp_mat)
    step_V_ = np.divide(SumUnoA + lam * SumV_A, temp_mat)
    
    V_ = np.multiply(V_, step_V_)
    
    return V_
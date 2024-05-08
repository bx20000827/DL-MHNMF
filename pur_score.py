import numpy as np
from scipy.stats import mode

def Purity(labels_true, labels_pred):
    labels_true = np.array(labels_true).flatten()
    labels_pred = np.array(labels_pred).flatten()
    total_purity = 0

    for i1 in np.unique(labels_pred):
        F = mode(labels_true[labels_pred == i1])[1][0]
        total_purity = total_purity + (F / np.sum(labels_pred == i1))

    purity = total_purity / len(np.unique(labels_pred))
    return purity

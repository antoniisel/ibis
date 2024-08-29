"""
Preprocessing
"""

from sklearn.preprocessing import MinMaxScaler


def preprocess_pbm(y):
    
    print("Preprocessing data for PBM")
    scaler_mn_mx = MinMaxScaler()
    y = scaler_mn_mx.fit_transform(y.reshape(-1, 1))

    return y

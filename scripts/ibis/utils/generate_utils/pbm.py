from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path




def pbm_gen_func(X, y, quantile=95):

    y = y.flatten()
    quantile_80 = np.percentile(y, quantile)

    indices = np.arange(len(y))
    filtered_indices_low = indices[y <= quantile_80]
    filtered_indices_high = indices[y > quantile_80]
    sample_size = len(filtered_indices_high)

    indices_low = np.random.choice(filtered_indices_low, size=sample_size, replace=False)
    mask = np.zeros(len(y))
    mask[indices_low] = 1
    mask[filtered_indices_high] = 1
    mask = mask.astype(bool)

    data = X[mask]
    labels = y.view(-1, 1)[mask]
    
    return data, labels




if __name__ == "__main__":
    X = torch.rand(size=(1000, 5, 35))
    y = torch.rand(size=(1000, 1))
    data, labels = pbm_gen_func(X, y)
    print(data.shape, labels.shape)
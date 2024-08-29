from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


def hts_gen_func(X, y):

    target_size = int((y == 1).sum())
    neg_size = int((y == 0).sum())
    neg_indices = np.arange(len(y))[y == 0]
    choice_size = min(neg_size, target_size)
    indices = np.random.choice(neg_indices, size=choice_size, replace=False)
    mask = np.zeros(len(y))
    mask[indices] = 1
    mask[np.arange(len(y))[y == 1]] = 1
    mask = mask.astype(bool)

    data = X[mask]
    labels = y[mask]
    
    return data, labels


if __name__ == "__main__":
    X = torch.rand(100, 5, 40)
    y = torch.zeros(100)
    y[:20] = 1
    data, labels = hts_gen_func(X, y)
    print(data.shape, labels.shape)
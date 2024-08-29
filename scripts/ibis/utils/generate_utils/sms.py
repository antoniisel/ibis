import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


def sms_gen_func(X, y):

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

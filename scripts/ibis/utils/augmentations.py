"""
Augmentations
"""

import numpy as np
import torch


def get_complement(seq, p=0.1): 
    k = np.random.randint(0, 100)
    if k < p*100:  
        seq[2, :], seq[1, : ]  = seq[1, : ], seq[2, : ] 
        seq[3, :], seq[4, : ]  = seq[4, : ], seq[3, : ]  
        
    return seq  

def flip(seq, p=0.1):
    k = np.random.randint(0, 100)
    if k < p*100:  
        return torch.flip(seq, dims=[1]) 


def get_reverse_compliment(seq, p=0.1): 
    k = np.random.randint(0, 100)
    if k < p*100:  
        seq[2, :], seq[1, : ]  = seq[1, : ], seq[2, : ] 
        seq[3, :], seq[4, : ]  = seq[4, : ], seq[3, : ]  
        return torch.flip(seq, dims=[1])   
     
    else:
        return seq



def mask_last_with_N(data, last_n=10, p=0.1):
    k = np.random.randint(0, 100)
    if k < p*100:            
        data[:, -last_n:] = 0 
        data[0, -last_n:] = 1  

    return data


def mask_first_and_last_with_N(data, max_n=120, p=0.1):
    k = np.random.randint(0, 100)
    if k < p*100:    
        n = np.random.randint(0, max_n)
        data[:, -n:] = 0 
        data[0, -n:] = 1 
        data[:, :n] = 0 
        data[0, :n] = 1   

    return data



def mask_N_with_position(data, max_n=100, p=0.1):
    i = np.random.randint(0, len(data))
    k = np.random.randint(0, 100)
    if k < p*100:  
        n = np.random.randint(0, max_n)          
        data[:, i:i+n] = 0 
        data[0, i:i+n] = 1  

    return data


def insert_N(data, n=3, p=0.1):
    k = np.random.randint(0, 100)
    if k < p*100:            
        indices = np.random.randint(0, 40, n)
        data[:, indices] = 0
        data[0, indices] = 1  

    return data


def insert_base(data, max_n=100, p=0.1, seq_size=35):
    k = np.random.randint(0, 100)
    if k < p*100:            
        indices = np.random.randint(0, seq_size, np.random.randint(0, max_n))
        data[:, indices] = 0
        data[np.random.randint(0, 5, size=len(indices)), indices] = 1  

    return data


def insert_N_w_size(data, n=7, p=0.1, seq_size=35):
    k = np.random.randint(0, 100)
    if k < p*100:            
        indices = np.random.randint(0, seq_size, n)
        data[:, indices] = 0
        data[0, indices] = 1  

    return data



if __name__ == "__main__":
    data = torch.zeros(size=(5, 40))
    data[0, ::2] = 1
    data[3, 1::2] = 1
    print(data)
    # data = torch.arange(9).view(3, 3)
    # print(torch.flip(data, dims=[1]))
    print(get_reverse_compliment(data))
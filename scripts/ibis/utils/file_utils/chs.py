from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import config


def read_chs_test(path):

    data, tags = read_fastq(path, start=1, skip=2, tags_read_params=[0, 2, 1, 1+7])
    print("Test data shape", tags.shape)
    return data, tags


def read_fastq(path, start=1, skip=4, tags_read_params=None):

    """
    Reads fastq file and
    returns train dataset of shape (num_samples, sequence_length, 5)
    where 5 corresponds to encoded ("N", "A", "T", "G", "C")
    """

    print("reading fasta file")
    sequences = np.array([])
    with open(path, "r", encoding='UTF-8') as file:
        lines = np.array(file.readlines())
        lines = np.apply_along_axis(np.vectorize(lambda x: x.strip()), -1, lines)    
        sequences = lines[start::skip]
        if tags_read_params:
            tags = [x[tags_read_params[2]:tags_read_params[3]] for \
                    x in lines[tags_read_params[0]::tags_read_params[1]]]
            


    sequences = np.char.replace(sequences, "N", "0")
    sequences = np.char.replace(sequences, "A", "1")
    sequences = np.char.replace(sequences, "T", "2")
    sequences = np.char.replace(sequences, "G", "3")
    sequences = np.char.replace(sequences, "C", "4")
    sequences = np.char.join(",", sequences)
    sequences = np.char.array(np.char.split(sequences, ','))
    sequences = np.array(sequences.astype(str), int)


    if tags_read_params:
        return np.eye(5)[sequences], np.array(tags)
    else: 
        return np.eye(5)[sequences]
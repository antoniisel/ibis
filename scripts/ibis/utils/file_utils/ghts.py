from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


def read_ghts_test(path):

    data, tags = read_fastq(path, start=1, skip=2, tags_read_params=[0, 2, 1, 1+7])
    print("Test data shape", data.shape)
    return data, tags


def read_ghts_train(method_path, target_name, test_size=0.2):


    np.random.seed(42)
    print("Reading GHTS data for", target_name)
    X, Y = [], []
    X_train, Y_train, X_test, Y_test = [], [], [], []

    X_target = method_path / f'X_{target_name}.npy'
    y_target =  method_path / f'y_{target_name}.npy'


    X.extend(np.load(X_target))
    Y.extend(np.load(y_target))
    target_size = len(Y)

    print("----"*30)
    print(f"Size for TF {target_name} as target =", target_size)
    print("----"*30)

    test_indices = np.random.choice(np.arange(len(Y)),
                                     size=int(target_size * test_size), replace=False, 
                                     )

    test_mask = np.zeros(len(Y))
    test_mask[test_indices] = 1
    test_mask = test_mask.astype(bool)

    X_test.extend(np.array(X)[test_mask])
    Y_test.extend(np.array(Y)[test_mask])
    X_train.extend(np.array(X)[~test_mask])
    Y_train.extend(np.array(Y)[~test_mask])
    print("Traing positive labels length:", len(Y_train))
    print("Testing positive labels length:", len(Y_test))
    print()


    TFs_x = sorted(list(x for x in method_path.iterdir() if not x.is_dir() and \
               target_name not in x.name and \
                 "X" in  x.name))  

    TFs_y  = sorted(list(y for y in method_path.iterdir() if not y.is_dir() and \
               target_name not in y.name and \
                 "y_" in y.name))  
    
    TF_num = len(TFs_y)

    for tfx, tfy in zip(TFs_x, TFs_y):
        tf_X = np.load(tfx)
        tf_Y = np.zeros_like(np.load(tfy))
        print(len(tf_Y))

        sample_size_max = target_size
        sample_size = min(len(tf_Y), sample_size_max)
        indices = np.random.choice(np.arange(len(tf_Y)), size=sample_size, replace=False)
        tf_X = np.array(tf_X)[indices]
        tf_Y = np.array(tf_Y)[indices]

        sample_size_max = target_size // TF_num
        sample_size = min(len(tf_Y), sample_size_max)
        test_indices = np.random.choice(np.arange(len(tf_Y)), 
                                        size=int(sample_size * test_size), replace=False)
        
        
        
        test_mask = np.zeros(len(tf_Y))
        test_mask[test_indices] = 1
        test_mask = test_mask.astype(bool)

        X_test.extend(np.array(tf_X)[test_mask])
        Y_test.extend(np.array(tf_Y)[test_mask])
        X_train.extend(np.array(tf_X)[~test_mask])
        Y_train.extend(np.array(tf_Y)[~test_mask])

    print("Traing labels length:", len(Y_train))
    print("Testing labels length:", len(Y_test))
    np.random.seed(None)
    return torch.tensor(np.array(X_train), dtype=torch.float32), \
            torch.tensor(np.array(Y_train), dtype=torch.float32), \
            torch.tensor(np.array(X_test), dtype=torch.float32), \
                torch.tensor(np.array(Y_test), dtype=torch.float32), \


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
    



if __name__ == "__main__":
    read_ghts_train(method_path=Path("/home/selivanov/ml_projects/ibis/ibis_model/data/train/GHTS"),
                    target_name="ZNF362")
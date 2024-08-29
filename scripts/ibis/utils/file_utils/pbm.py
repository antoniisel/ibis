
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from os import listdir
from os.path import isfile, join
from pathlib import Path


def read_pbm_train(method_path, target_name, test_size=0.2, quantile=95) -> np.array:

    print("Read train pbm data")
    dir_path = method_path / target_name
    files = sorted([f for f in listdir(dir_path) if isfile(join(dir_path, f)) and not f.startswith("SD")])
    data = pd.read_csv(dir_path / files[0], sep='\t')
    sequences = np.array(data['pbm_sequence'])
    sequences = np.apply_along_axis(np.vectorize(lambda x: x.strip()), -1, sequences) 
    intensity = np.array(data['mean_signal_intensity'])
    sequences = np.char.replace(sequences, "N", "0")
    sequences = np.char.replace(sequences, "A", "1")
    sequences = np.char.replace(sequences, "T", "2")
    sequences = np.char.replace(sequences, "G", "3")
    sequences = np.char.replace(sequences, "C", "4")
    sequences = np.char.join(",", sequences)
    sequences = np.char.array(np.char.split(sequences, ','))
    sequences = np.array(sequences.astype(str), int)
    sequences = np.eye(5)[sequences]

    scaler = MinMaxScaler(feature_range=(0, 1))
    intensity = scaler.fit_transform(np.array(intensity).reshape(-1, 1)).flatten()


    
    np.random.seed(42)


    X_train, Y_train, X_test, Y_test = [], [], [], []
    quantile_80 = np.percentile(intensity, quantile)
    print(f"quantile{quantile}=",quantile_80)


    filtered_data_high = sequences[intensity > quantile_80]
    filtered_intensity_high = intensity[intensity > quantile_80]
    sample_size = int(len(filtered_data_high) * test_size)
    test_indices = np.random.choice(np.arange(len(filtered_data_high)), size=sample_size, replace=False)
    test_mask = np.zeros(len(filtered_data_high))
    test_mask[test_indices] = 1
    test_mask = test_mask.astype(bool)

    X_test.extend(filtered_data_high[test_mask])
    Y_test.extend(filtered_intensity_high[test_mask])
    X_train.extend(filtered_data_high[~test_mask])
    Y_train.extend(filtered_intensity_high[~test_mask])


    filtered_data_low = sequences[intensity <= quantile_80]
    filtered_intensity_low = intensity[intensity <= quantile_80]
    test_indices = np.random.choice(np.arange(len(filtered_data_low)), size=sample_size, replace=False)
    test_mask = np.zeros(len(filtered_data_low))
    test_mask[test_indices] = 1
    test_mask = test_mask.astype(bool)

    X_test.extend(filtered_data_low[test_mask])
    Y_test.extend(filtered_intensity_low[test_mask])
    X_train.extend(filtered_data_low[~test_mask])
    Y_train.extend(filtered_intensity_low[~test_mask])

    np.random.seed(None)

    print("Data size:", "train:", len(X_train), "test:", len(X_test))

    return torch.tensor(np.array(X_train), dtype=torch.float32), \
            torch.tensor(np.array(Y_train).reshape(-1, 1), dtype=torch.float32), \
            torch.tensor(np.array(X_test), dtype=torch.float32), \
                torch.tensor(np.array(Y_test).reshape(-1, 1), dtype=torch.float32), \




def read_pbm_test(path:str, one_hot=False) -> np.array:

    print("Read test pbm data")
    sequences = np.array([])
    tags = np.array([])
    with open(path, "r", encoding='UTF-8') as file:
        lines = np.array(file.readlines())
        lines = np.apply_along_axis(np.vectorize(lambda x: x.strip()), -1, lines)    
        sequences = lines[1::2]
        tags = np.apply_along_axis(np.vectorize(lambda x: x.lstrip('>')), -1, lines[0::2])   
        linkers = np.apply_along_axis(np.vectorize(lambda x: x.split(';')[-1]), -1, tags)   
        tags = np.apply_along_axis(np.vectorize(lambda x: x.split(' ')[0]), -1, tags)  
        linkers = np.apply_along_axis(np.vectorize(lambda x: x[7:]), -1, linkers)  
    sequences = [sequence[len(linker):] for sequence, linker in zip(sequences, linkers)]

    sequences = np.char.replace(sequences, "N", "0")
    sequences = np.char.replace(sequences, "A", "1")
    sequences = np.char.replace(sequences, "T", "2")
    sequences = np.char.replace(sequences, "G", "3")
    sequences = np.char.replace(sequences, "C", "4")
    sequences = np.char.join(",", sequences)
    sequences = np.char.array(np.char.split(sequences, ','))
    sequences = np.array(sequences.astype(str), int)
    sequences = np.eye(5)[sequences]
    
    return sequences, tags



if __name__ == "__main__":
    x_tr, y_tr, x_ts, y_ts = read_pbm_train(Path("/home/selivanov/ml_projects/ibis/ibis_model/data/train/PBM"), "LEF1")
    print(x_tr.shape, x_ts.shape, y_ts.shape)
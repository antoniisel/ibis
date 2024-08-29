from ibis.dataset import ibisDataset
from ibis.utils.general import load_model_weights
import config
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np


from ibis.predict import predict_g2a
from ibis.configurate import get_test_configuration

from pathlib import Path




def g2a_single_TF_prediction(target_method, source_method, target_TF, training_tags):

    read_func, weights_dir_load_g2a, test_path, predictions_dir, model_g2a = get_test_configuration(method_name=source_method, target_TF=target_TF, train_tags=training_tags)
    read_func_g2a, weights_dir_load, test_path_g2a, predictions_dir_g2a, model = get_test_configuration(method_name=target_method, target_TF=target_TF, train_tags=training_tags)


    test_data, tags = read_func_g2a(test_path_g2a)

    test_data = torch.tensor(test_data, dtype=torch.float32).permute(0, 2, 1)

    print(test_data.shape)
    y = torch.ones((len(test_data), 1))

    print(weights_dir_load_g2a)
    device = config.device
    model_g2a = load_model_weights(model_g2a, weights_dir_load_g2a)
    model_g2a = model_g2a.to(config.device)

    batch_size = 1000
    
    print("Device", device)

    predictions = predict_g2a(model_g2a, test_data, target_TF, device)


    predictions_dir_g2a.mkdir(parents=True, exist_ok=True)
    np.save(predictions_dir_g2a / "predictions.npy", predictions)
    np.save(predictions_dir_g2a / "tags.npy", tags)



if __name__ == "__main__":
    # for target_TF in ["GABPA", "PRDM5", "ZNF362", "ZNF407"]:
    for target_TF in ["NFKB1"]:
        g2a_single_TF_prediction(target_method="HTS", source_method="GHTS", target_TF=target_TF, training_tags=["first_final"]
                                )
        


        
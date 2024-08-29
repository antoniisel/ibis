from ibis.utils.general import load_model_weights
import config
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np


from ibis.predict import predict_genome
from ibis.configurate import get_test_configuration

from pathlib import Path


def a2g_single_TF_prediction(target_method, source_method, target_TF, training_tags, slice_size, step_size):

    print("single_TF_prediction", target_TF, source_method)
    read_func, weights_dir_load_a2g, test_path, predictions_dir, model_a2g = get_test_configuration(method_name=source_method, target_TF=target_TF, train_tags=training_tags)
    read_func_a2g, weights_dir_load, test_path_a2g, predictions_dir_a2g, model = get_test_configuration(method_name=target_method, target_TF=target_TF, train_tags=training_tags)


    test_data, tags = read_func_a2g(test_path_a2g)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    # test_data_reverse = torch.flip(test_data, dims=[2])

    print(test_data.shape)
    y = torch.ones((len(test_data), 1))

    print(weights_dir_load_a2g)
    device = config.device
    model_a2g = load_model_weights(model_a2g, weights_dir_load_a2g)
    model_a2g = model_a2g.to(config.device)

    batch_size = 1000
    
    print("Device", device)

    predictions = predict_genome(model_a2g, test_data, target_TF, device, slice_size, step_size)

    # predictions_reverse = predict_genome(model, test_data_reverse, target_TF, device, slice_size, step_size)


    predictions_dir_a2g.mkdir(parents=True, exist_ok=True)
    np.save(predictions_dir_a2g / "predictions.npy", predictions)
    np.save(predictions_dir_a2g / "tags.npy", tags)



if __name__ == "__main__":
    # for target_TF in ["LEF1", "NACC2", "RORB", "TIGD3"]:
    for target_TF in ["NFKB1"]:
        a2g_single_TF_prediction(target_method="GHTS", source_method="HTS", target_TF=target_TF, training_tags=["first_final",], 
                                 slice_size=40, step_size=20)
        


        
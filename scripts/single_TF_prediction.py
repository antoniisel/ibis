from ibis.configurate import get_test_configuration
from ibis.dataset import ibisDataset
from ibis.utils.general import load_model_weights
import config
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from ibis.predict import predict




def single_TF_prediction(method_name, target_TF, training_tags=["default",]):

    read_func, weights_dir_load, test_path, predictions_dir, model = get_test_configuration(method_name, target_TF, training_tags)

    test_data, tags = read_func(test_path)
    test_data = torch.tensor(test_data, dtype=torch.float32).permute(0, 2, 1)

    y = torch.ones((len(test_data), 1))


    device = config.device
    print(weights_dir_load)
    model = load_model_weights(model, weights_dir_load)
    model = model.to(config.device)

    batch_size = 1000
    
    print("Device", device)

    test_data = ibisDataset(test_data, y, data_gen_func=None, augmentations=None)
    test_loader =  DataLoader(test_data, batch_size=batch_size, shuffle=False)

    predictions = predict(model, test_loader, device=device)



    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.save(predictions_dir / "predictions.npy", predictions)
    np.save(predictions_dir / "tags.npy", tags)



if __name__ == "__main__":
    single_TF_prediction("HTS", "NFKB1", training_tags=["first_final"])

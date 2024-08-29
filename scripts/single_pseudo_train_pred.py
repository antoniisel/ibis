from ibis.train import train
from ibis.configurate import get_train_configuration, get_test_configuration
from ibis.dataset import ibisDataset
from ibis.utils.augmentations import insert_N, mask_last_with_N, insert_N_w_size, \
                                                    mask_N_with_position, get_reverse_compliment,\
                                                    get_complement, get_reverse_compliment, \
                                                         get_reverse_compliment,\
                                                            flip, insert_base, mask_N_with_position

from ibis.utils.general import load_model_weights
import config
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
from pathlib import Path
import pandas as pd
from single_TF_prediction import predict
from torch.utils.data import DataLoader
from submission import submit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from ibis.utils.file_utils.ghts import read_ghts_test
from ibis.utils.file_utils.hts import read_hts_test
from ibis.models.resnet_18g import resnet18_1d
from ibis.models.resnet_34g import resnet34_1d_binary
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from tqdm import tqdm


def bmc_loss(pred, target, noise_var):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var)
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)


def train_folds(data, labels, weights_path, iter_num):

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    predictions = 0
    # Iterate through each fold
    for i, (train_index, val_index) in tqdm(enumerate(kf.split(data)), desc="cross_val tqdm"):
            weights_path_save = weights_path / f"iter{iter_num}" / f"split{i}"
            weights_path_save.mkdir(exist_ok=True, parents=True)
            weights_dir_save = weights_path_save / "best_ResNet18.pth"
            X_train, X_val = data[train_index], data[val_index]
            y_train, y_val = labels[train_index], labels[val_index]
            X_train, X_test, y_train,  y_test = map(lambda x: torch.tensor(x, dtype=torch.float32), (X_train, X_val, np.array(y_train),  np.array(y_val)))

            print("Success, path to save weights:", weights_dir_save)
            print()
            print("Start data reading" )
            X_train, X_test =  X_train.permute(0, 2, 1), X_test.permute(0, 2, 1)
            print("Success")
            print()

            model = resnet18_1d()
            train_data = ibisDataset(X_train, y_train, data_gen_func=None, augmentations=None)
            test_data = ibisDataset(X_test, y_test,data_gen_func=None, augmentations=None)
            init_noise_sigma = 8.0
            sigma_lr = 1e-2
            criterion = BMCLoss(init_noise_sigma)
            optimizer = optim.Adam(model.parameters())
            optimizer.add_param_group({'params': criterion.noise_sigma, 
                                            'lr': sigma_lr, 'name': 'noise_sigma'})


            model.to(config.device)
            num_epochs = config.num_epochs

            batch_size = config.batch_size
            device = config.device
            print("Device", device)

            train(model, 
                    optimizer,
                    criterion,
                    num_epochs,
                    train_data,
                    test_data,
                    device,
                    batch_size,
                    True,
                    weights_dir_save)

            batch_size = 1000
            
            test_data = ibisDataset(torch.tensor(data, dtype=torch.float32).permute(0, 2, 1), labels, data_gen_func=None, augmentations=None)
            test_loader =  DataLoader(test_data, batch_size=batch_size, shuffle=False)
            model = resnet18_1d().to(device)
            model = load_model_weights(model, weights_dir_save).to(device)
            predictions += predict(model, test_loader, device=device)

    predictions = predictions / 3

    return predictions



def pseudo_training(method_name, target_TF, training_tags):

    read_func, weights_dir_load, test_path, predictions_dir, model = get_test_configuration(method_name, target_TF, training_tags)  
    initial_pred_path = predictions_dir.parent.parent / \
                                (method_name + "_" + "_" + \
                                "_".join(training_tags) +"_hts_a2g" + ".tsv")

    df = pd.read_csv(initial_pred_path, delimiter="\t")
    test_data = read_ghts_test(test_path)
    labels = df[target_TF]
    data = test_data[0]
    print("data shape:", data.shape)

    weights_path = Path(f"/home/selivanov/ml_projects/ibis/final_ibis/weights/experiments/{method_name}/{target_TF}")
    iter_nums = ["1", "2"]

    predictions = labels
    for iter_num in tqdm(iter_nums):
        predictions = train_folds(data, predictions, weights_path, iter_num)

    np.save(predictions_dir / "predictions_pseudo.npy", predictions) 





if __name__ == "__main__":
    pseudo_training("GHTS", "NFKB1", training_tags=["first_final"],
                      )












# predict_df.to_csv(f"/home/selivanov/ml_projects/ibis/final_ibis/data/predictions/GHTS/ \
#                   {method_name}_pseudo_{"_".join(TFs)}_iter{iter_num}.tsv", 
#                   sep='\t', index=False)
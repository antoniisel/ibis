from ibis.train import train
from ibis.configurate import get_train_configuration
from ibis.dataset import ibisDataset
from ibis.utils.augmentations import insert_N, mask_last_with_N, insert_N_w_size, \
                                                    mask_N_with_position, get_reverse_compliment,\
                                                    get_complement, get_reverse_compliment, \
                                                         get_reverse_compliment,\
                                                            flip

from ibis.utils.general import load_model_weights
import config
import torch.optim as optim
from timm.optim.adamp import AdamP
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss



def single_TF_training(method_name, target_TF, augmentations=None, training_tags=["default",]):
    print("single_TF_training", target_TF)
    print("Start data configuration for", method_name, target_TF)


    gen_func, read_func, criterion, \
        method_path, weights_dir_save, weights_dir_load, \
            is_regression, model = get_train_configuration(method_name, target_TF, training_tags)
    
    print("Success, path to save weights:", weights_dir_save)
    print()
    print("Start data reading" )
    X_train, y_train, X_test, y_test = read_func(method_path, target_TF)
    X_train, X_test =  X_train.permute(0, 2, 1), X_test.permute(0, 2, 1)
    print("Success")
    print()
    
    train_data = ibisDataset(X_train, y_train, gen_func, augmentations)
    test_data = ibisDataset(X_test, y_test, gen_func, augmentations=None)


    model = load_model_weights(model, weights_dir_load)

    if method_name == "PBM":
        init_noise_sigma = 8.0
        sigma_lr = 1e-2
        criterion = BMCLoss(init_noise_sigma)
        optimizer = optim.Adam(model.parameters())
        optimizer.add_param_group({'params': criterion.noise_sigma, 
                                   'lr': sigma_lr, 'name': 'noise_sigma'})
        
    else:
        optimizer = optim.Adam(model.parameters())

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
            is_regression,
            weights_dir_save)



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
    

if __name__ == "__main__":
    single_TF_training("HTS", "NFKB1", training_tags=["first_final"],
                        augmentations=None,
                       )
    
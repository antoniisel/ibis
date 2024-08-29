from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path



def load_model_weights(model, weights_dir_load):

    if weights_dir_load.exists():
        print("Loading weights")
        model.load_state_dict(torch.load(weights_dir_load))
    else:
        print("Weights are not found")
    print("----"*30)

    return model


def save_weights(model, weights_dir_save):

    
    weights_dir_save.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), weights_dir_save)


def add_val_metrics_to_writer(writer, epoch, epoch_loss, accuracy, precision, recall, f1, roc_auc):
    writer.add_scalar('Metrics/Accuracy', accuracy, epoch)
    writer.add_scalar('Metrics/Precision', precision, epoch)
    writer.add_scalar('Metrics/Recall', recall, epoch)
    writer.add_scalar('Metrics/F1', f1, epoch)
    writer.add_scalar('Metrics/ROC_AUC', roc_auc, epoch)






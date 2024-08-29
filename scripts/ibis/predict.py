from ibis.dataset import ibisDataset
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

def predict(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        all_outputs = []
        for inputs, labels in tqdm(test_loader, desc=f"Prediction"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            preds = outputs.cpu().detach().numpy()
            all_outputs.extend(preds)

    return np.array(all_outputs)



def predict_genome(model, data, TF_name, device, slice_size=40, step_size=20):
    """
    data.shape : (num_samples, 301, 5) 
    """

    num_samples = data.shape[0] 
    seq_len = data.shape[1] 
    num_steps = (seq_len - slice_size) // step_size + 2

    print(num_samples, seq_len, num_steps, slice_size, step_size)

    # num_to_add = step_size - (seq_len - slice_size) % step_size
    # add_N = [1, 0, 0, 0, 0] * num_samples * num_to_add
    # add_N = torch.tensor(add_N).view(num_samples, num_to_add, 5)
    # data = torch.cat((data, add_N), dim=1)


    total = torch.zeros(size=(num_samples, num_steps))
    batch_size = 10000

    model.to(device)
    model.eval()
    with torch.no_grad(): 
        for j in tqdm(range(0, len(data), batch_size), desc=f"Testing {TF_name}"): 
            data_batch = data[j:j+batch_size].to(device)
            for i in range(num_steps):     
                if i == num_steps - 1:
                # if False:
                    data_slice = data_batch[:,-slice_size:,:] 
                    # print(i, -slice_size) 
                    # print(data_slice)     
                else:
                    # print(i, i*step_size, i*step_size+slice_size)
                    data_slice = data_batch[:,i*step_size:i*step_size+slice_size,:]
                preds = model(data_slice.permute(0, 2, 1))
                total[j:j+batch_size:, i] = preds.view(-1)
        print(total[0])
        res, _ = total.max(dim=1)
        # res = total.mean(dim=1)
        
    return res


def predict_g2a(model, data, TF_name, device, seq_size=301):

    num_samples = data.shape[0] 
    seq_len = data.shape[2] 

    n = seq_size - seq_len
    long_data = torch.zeros(size=(num_samples, 5, seq_size))
    n_left = n // 2
    n_right = n - n // 2
    long_data[:, :, n_left:-n_right] = data
    long_data[:, 0, :n_left] = 1
    long_data[:, 0, n_right:] = 1
    y = torch.ones(num_samples)
    batch_size = 10000


    test_data = ibisDataset(long_data, y, data_gen_func=None, augmentations=None)
    test_loader =  DataLoader(test_data, batch_size=batch_size, shuffle=False)

    predictions = predict(model, test_loader, device=device)

    model.to(device)
    model.eval()

    return predictions




if __name__ == "__main__":
    data = torch.zeros(100, 5, 40)
    predict_g2a(1, data, 1, 1)
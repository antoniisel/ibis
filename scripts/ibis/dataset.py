import numpy as np
from torch.utils.data import TensorDataset, Dataset
import torch


HTSDataset = TensorDataset

class ibisDataset(Dataset):

  def __init__(self, X, y, data_gen_func, data_clean_func=None, augmentations=None) -> None:
    self.data_gen_func = data_gen_func
    self.X = X
    self.Y = y
    self.X_clean = X
    self.Y_clean = y
    self.data = self.X_clean
    self.labels = self.Y_clean
    self.data_clean_func = data_clean_func
    self.augmentations = augmentations

  def __len__(self) -> int:
    return len(self.labels)

  def __getitem__(self, idx: int):
    data, label = self.data[idx], self.labels[idx]
    if self.augmentations:
       for aug in self.augmentations:
            data = aug(data)

    return data, label

  def generate_data(self) -> None:
    """Generates data by calling the data_gen_func."""
    if self.data_gen_func:
      print("Start data generating")
      self.data, self.labels = self.data_gen_func(self.X_clean, self.Y_clean)
      print(f"Data generated for epoch, data shape:",self.data.shape)


  def clean_data(self) -> None:
    if self.data_clean_func:
      indices = np.arange(len(self.X))
      np.random.shuffle(indices)
      self.X = self.X[indices]
      self.Y = self.Y[indices]
      print("Start data cleaning")
      self.X_0 = torch.cat(tuple(self.X[self.Y==0]), dim=1)[:,:(self.X[self.Y==0].shape[0] // (5*301))*(5*301)].permute(1, 0).reshape(-1, 301, 5).permute(0, 2, 1)
      self.X_1 = torch.cat(tuple(self.X[self.Y==1]), dim=1)[:,:(self.X[self.Y==1].shape[0] // (5*301))*(5*301)].permute(1, 0).reshape(-1, 301, 5).permute(0, 2, 1)
      self.X_clean =  torch.cat((self.X_0, self.X_1))
      self.Y_clean = torch.cat((torch.zeros(self.X_0.shape[0]), torch.ones(self.X_1.shape[0])))

      for i in range(10):
        print(i)
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.Y = self.Y[indices]
        print("Start data cleaning")
        self.X_0 = torch.cat(tuple(self.X[self.Y==0]), dim=1)[:,:(self.X[self.Y==0].shape[0] // (5*301))*(5*301)].permute(1, 0).reshape(-1, 301, 5).permute(0, 2, 1)
        self.X_1 = torch.cat(tuple(self.X[self.Y==1]), dim=1)[:,:(self.X[self.Y==1].shape[0] // (5*301))*(5*301)].permute(1, 0).reshape(-1, 301, 5).permute(0, 2, 1)
        self.X_clean =  torch.cat((torch.cat((self.X_0, self.X_1)), self.X_clean))
        self.Y_clean = torch.cat((torch.zeros(self.X_0.shape[0]), torch.ones(self.X_1.shape[0]), self.Y_clean))


      
      print(f"Data cleaned for epoch, data shape:",self.X_clean.shape)
      self.data = self.X_clean
      self.labels = self.Y_clean




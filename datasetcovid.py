import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class CovidDataset(Dataset):
    def __init__(self, path):
        #Dataloading
        xy=np.loadtxt(path,delimiter=",",dtype=np.float32,skiprows=1)
        self.x=torch.Tensor(xy[:,:-1])
        self.y=torch.Tensor(xy[:,-1])
        self.n_samples=xy.shape[0]
    def __getitem__(self,index):
        #DatasetAllocation
        return self.x[index],self.y[index]
    def __len__(self):
        #DatasetLength
        return self.n_samples

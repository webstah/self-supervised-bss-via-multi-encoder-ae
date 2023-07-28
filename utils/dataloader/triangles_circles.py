import os
from hydra.utils import get_original_cwd
import pickle
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


def min_max(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        data_path = os.path.join(get_original_cwd(), config.data_path)
        with open(data_path, 'rb') as outfile:
            data = pickle.load(outfile, encoding='latin1')
        
        self.train_split, self.val_split = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
        self.x_plot = [torch.tensor(min_max(im[0]), dtype=torch.float32).permute(2, 0, 1) for im in self.val_split[:8]]

    def setup(self, stage):
        self.train_data = Dataset(self.train_split)
        self.val_data = Dataset(self.val_split)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=False)

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, c, t = self.data[idx]
        x, c, t = min_max(x), min_max(c), min_max(t)
        x = torch.tensor(x, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)

        return x.permute(2, 0, 1), c.permute(2, 0, 1), t.permute(2, 0, 1)
    
    
        
    
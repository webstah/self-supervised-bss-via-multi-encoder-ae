import os
import pickle
from glob import glob
from hydra.utils import get_original_cwd

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        data_path = config.data_path

        data_paths = glob(os.path.join(get_original_cwd(), data_path, 'train_0/*.dat'))
        
        train_data_paths = data_paths[:450]
        train_data = []
        for train_data_path in tqdm(train_data_paths):
            with open(train_data_path, 'rb') as train_outfile:
                train_data = train_data + pickle.load(train_outfile, encoding='latin1')
                
        self.train_split = train_data
        
        print('Train:', len(self.train_split))
        
        val_data_paths = data_paths[450:]
        print(len(val_data_paths))
        val_data = []
        for val_data_path in tqdm(val_data_paths):
            with open(val_data_path, 'rb') as val_outfile:
                val_data = val_data + pickle.load(val_outfile, encoding='latin1')
        
        self.val_split = val_data
        
        self.x_plot = val_data[0:5]

    def setup(self, stage):
        self.train_data = ECGDataset(self.train_split)
        self.val_data = ECGDataset(self.val_split)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=False)
        
class ECGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def _local_normalize(self, x):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        
        return x

    def __getitem__(self, idx):
        x_, y_ = self.data[idx][0], self.data[idx][1]
        
        x = torch.tensor(x_).squeeze()
        x = self._local_normalize(x).float().unsqueeze_(0)
        
        y = torch.tensor(y_).squeeze()
        y = self._local_normalize(y).float().unsqueeze_(0)

        return x, y
    
    
        
    
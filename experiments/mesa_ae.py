import os

import torch
from torch import nn
import pytorch_lightning as pl

from utils.importer import import_model_from_config
from utils.plots import plot_signal

class Experiment(pl.LightningModule):
    def __init__(self, config, x_plot=None):
        super().__init__()

        self.hidden = config.hidden
        
        self.lr = config.lr
        self.lr_step_size = config.lr_step_size
        self.weight_decay = config.weight_decay
        
        self.x_plot = x_plot
        
        self.save_plots = config.save_plots
        self.plot_dir = config.plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.plot_step = config.plot_step
        
        model_import = import_model_from_config(config)
        self.model = model_import(input_channels=config.input_channels, channels=config.channels, 
                                  hidden=config.hidden)
        
        self.prepare_data()

        self.loss = nn.BCEWithLogitsLoss()
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, x): 
        pred, _ = self.model(x)

        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        x_pred = self.model(x)
        loss = self.loss(x_pred, y)
        self.log(f'recon_loss/train', loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
        
        # TODO: log cosine alignment
        self.log(f'resp_cos_sim/train', torch.mean(self.cosine_sim(torch.sigmoid(x_pred.detach()), y)), on_step=True, 
                                            on_epoch=True, prog_bar=True)
                
        if batch_idx % 20 == 0:
            plot = plot_signal(y, torch.sigmoid(x_pred))
            self.logger.experiment.add_image('images/train_plot', plot, batch_idx)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        x_pred = self.model(x)
        loss = self.loss(x_pred, y)
        self.log(f'recon_loss/val', loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
        self.log(f'resp_cos_sim/val', torch.mean(self.cosine_sim(torch.sigmoid(x_pred.detach()), y)), on_step=True, 
                                            on_epoch=True, prog_bar=True)
                
        if batch_idx % 5 == 0:
            plot = plot_signal(y, torch.sigmoid(x_pred))
            self.logger.experiment.add_image('images/val_plot', plot, batch_idx)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=0.1, last_epoch=-1),
            'name': 'step_lr_scheduler',
         }
        
        return [optimizer], [scheduler]
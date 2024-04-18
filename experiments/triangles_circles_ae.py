import torch
from torch import nn

import pytorch_lightning as pl

from utils.importer import import_model_from_config
from utils.plots import plot_grid

class Experiment(pl.LightningModule):
    def __init__(self, config, x_plot=None):
        super().__init__()
        
        self.hidden = config.hidden
        
        self.lr = config.lr
        self.lr_step_size = config.lr_step_size
        self.weight_decay = config.weight_decay
        
        self.prepare_data()

        model_import = import_model_from_config(config)
        self.model = model_import(input_channels=config.input_channels, 
                                  output_channels=config.output_channels, 
                                  channels=config.channels, 
                                  hidden=config.hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x): 
        pred = self.model(x)

        return pred

    def training_step(self, batch, batch_idx):
        x, c, t = batch

        x_pred = self.model(x)
        recon_loss = (self.loss(x_pred[:, 0].unsqueeze(1), c) + self.loss(x_pred[:, 1].unsqueeze(1), t))/2
        self.log(f'recon_loss/train', recon_loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
        if batch_idx % 50 == 0:
            plot = plot_grid(torch.sigmoid(x_pred[:, 0].detach()).unsqueeze(1), x_pred[:, 1].detach().unsqueeze(1))
            self.logger.experiment.add_image('images/train_plot', plot, batch_idx)
            
        return recon_loss
    
    def validation_step(self, batch, batch_idx):
        x, c, t = batch

        x_pred = self.model(x)
        recon_loss = (self.loss(x_pred[:, 0].unsqueeze(1), c) + self.loss(x_pred[:, 1].unsqueeze(1), t))/2
        
        self.log(f'recon_loss/val', recon_loss, on_step=False, 
                                on_epoch=True, prog_bar=True)
        if batch_idx % 50 == 0:
            plot = plot_grid(torch.sigmoid(x_pred[:, 0].detach()).unsqueeze(1), x_pred[:, 1].detach().unsqueeze(1))
            self.logger.experiment.add_image('images/val_plot', plot, batch_idx)
        
        return recon_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=0.1, last_epoch=-1),
            'name': 'step_lr_scheduler',
         }
        
        return [optimizer], [scheduler]
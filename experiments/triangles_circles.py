import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import torch
from torch import nn

import pytorch_lightning as pl

from utils.importer import import_model_from_config
from utils.plots import plot_grid

from models.separation_loss import WeightSeparationLoss

class Experiment(pl.LightningModule):
    def __init__(self, config, x_plot=None):
        super().__init__()
        self.automatic_optimization = False
        
        self.hidden = config.hidden
        self.num_encoders = config.num_encoders
        
        self.lr = config.lr
        self.lr_step_size = config.lr_step_size
        self.weight_decay = config.weight_decay
        
        self.sep_loss = config.sep_loss
        self.sep_lr = config.sep_lr
        
        self.z_decay = config.z_decay
        
        self.zero_loss = config.zero_loss
        self.zero_lr = config.zero_lr
        
        self.x_plot = x_plot
        
        self.save_plots = config.save_plots
        self.plot_dir = config.plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.plot_step = config.plot_step
        
        self.prepare_data()

        model_import = import_model_from_config(config)
        self.model = model_import(input_channels=config.input_channels, image_hw=config.image_hw, channels=config.channels, 
                                  hidden=config.hidden, use_weight_norm=config.use_weight_norm,
                                  num_encoders=config.num_encoders, norm_type=config.norm_type)
        self.loss = nn.BCEWithLogitsLoss()
        self.separation_loss = WeightSeparationLoss(config.num_encoders, config.sep_norm)

    def forward(self, x): 
        pred = self.model(x)

        return pred

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        x, c, t = batch

        x_pred, z = self.model(x)
        recon_loss = self.loss(x_pred, x)
        self.log(f'recon_loss/train', recon_loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
        
        loss = recon_loss
        for z_i in z:
            loss = loss + self.z_decay * torch.mean(z_i**2)
        
        if self.sep_loss:
            sep_loss = self.separation_loss(self.model.decoder)
            self.log(f'sep_loss/train', sep_loss, on_step=False, 
                                    on_epoch=True, prog_bar=True)
            
            loss = loss + sep_loss*self.sep_lr
        
        if self.zero_loss:
            z_zeros = [torch.zeros(x.shape[0], self.hidden//self.num_encoders, z[0].shape[-1], z[0].shape[-1]).to(self.device) for _ in range(self.num_encoders)]
            x_pred_zeros = self.model.decode(z_zeros, True)
            zero_recon_loss = self.loss(x_pred_zeros, torch.zeros_like(x_pred_zeros))
            loss = loss + zero_recon_loss*self.zero_lr
            self.log(f'zero_recon_loss/train', zero_recon_loss, on_step=False, 
                                on_epoch=True, prog_bar=True)
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val=None)
        optimizer.step()

        if batch_idx % 50 == 0:
            plot = plot_grid(x, x_pred)
            self.logger.experiment.add_image('images/train_plot', plot, batch_idx)
        
        if batch_idx % self.plot_step == 0 and self.save_plots:
            self.save_inference_samples(self.x_plot)
            self.save_weight_visualizations()

    
    def validation_step(self, batch, batch_idx):
        x, c, t = batch

        x_pred, z = self.model(x)
        recon_loss = self.loss(x_pred, x)
        
        self.log(f'recon_loss/val', recon_loss, on_step=False, 
                                on_epoch=True, prog_bar=True)
        if batch_idx % 50 == 0:
            plot = plot_grid(x, x_pred)
            self.logger.experiment.add_image('images/val_plot', plot, batch_idx)
        
        return recon_loss
    
    def save_inference_samples(self, x):
        step = self.global_step
        title = f'Step: {step}'
        save_name = os.path.join(self.plot_dir, f'images_{step}.png')
        x = [x_i.to(self.device) for x_i in x]
        x = torch.stack(x, dim=0).type(torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x)
            y = []
            for z_a in range(len(z)):
                for z_b in range(len(z)):
                    if z_a != z_b:
                        z_pad = [torch.zeros_like(z[0]) for _ in range(len(z))]
                        z_pad[z_a] = z[z_a]
                        z_pad[z_b] = z[z_b]
                        y_i = self.model.decode(z_pad)
                        y_i = torch.sigmoid(y_i).permute(0, 2, 3, 1).cpu().numpy()
                        y.append(y_i)
                    else:
                        z_pad = [torch.zeros_like(z[0]) for _ in range(len(z))]
                        z_pad[z_a] = z[z_a]
                        y_i = self.model.decode(z_pad)
                        y_i = torch.sigmoid(y_i).permute(0, 2, 3, 1).cpu().numpy()
                        y.append(y_i)

        # display pairs of embeddings
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(title)
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(len(z), len(z)),
                        axes_pad=0.02,
                        )

        for ax, im in zip(grid, y):
            im = im.astype(np.float32)
            ax.axis('off')
            ax.set_ymargin(50)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.imshow(im[1], cmap='gray')

        plt.savefig(save_name)
        plt.close()
        self.model.train()
    
    def save_weight_visualizations(self):
        step = self.global_step
        title = f'Step: {step}'
        save_name = os.path.join(self.plot_dir, f'weights_{step}.png')

        weights = []
        shapes = []
        for name, w in self.model.decoder.named_parameters():
            if name.split('.')[-1] == 'weight' and len(w.shape) == 4:
                weights.append(torch.sum(torch.sum(torch.abs(w.detach()), dim=-1), dim=-1).cpu().numpy())
                shapes.append(tuple(w.shape))
        
        fig, axs = plt.subplots(1, len(weights), figsize=(14,4.25))
        fig.suptitle(title)
        fig.tight_layout()
        for ax, weights, title in zip(axs, weights, shapes):
            ax.set_title(title)
            ax.axis('off')
            ax.imshow(weights, aspect='auto')
            
        plt.savefig(save_name)
        plt.close()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=0.1, last_epoch=-1),
            'name': 'step_lr_scheduler',
         }
        
        return [optimizer], [scheduler]
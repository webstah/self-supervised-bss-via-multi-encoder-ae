import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import torch
from torch import nn

import pytorch_lightning as pl

from utils.importer import import_model_from_config

from models.separation_loss import WeightSeparationLossAlternative
from utils.plots import plot_signal, z_norm

import matplotlib

class Experiment(pl.LightningModule):
    def __init__(self, config, x_plot=None):
        super().__init__()
        
        self.hidden = config.hidden
        self.num_encoders = config.num_encoders
        
        self.input_signal_type = config.input_signal_type
        
        self.lr = config.lr
        self.lr_step_size = config.lr_step_size
        self.weight_decay = config.weight_decay
        
        self.sep_loss = config.sep_loss
        self.sep_lr = config.sep_lr
        
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
        self.model = model_import(input_channels=config.input_channels, input_length=config.input_length, channels=config.channels, 
                                  hidden=config.hidden, use_weight_norm=config.use_weight_norm,
                                  num_encoders=config.num_encoders, norm_type=config.norm_type,
                                  input_padding=config.input_padding)

        self.loss = nn.BCEWithLogitsLoss()
        self.separation_loss = WeightSeparationLossAlternative(config.num_encoders, config.sep_norm)

    def forward(self, x): 
        pred, _ = self.model(x)

        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch

        x_pred, z = self.model(x)
        recon_loss = self.loss(x_pred, x)
        self.log(f'recon_loss/train', recon_loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
        loss = recon_loss
        
        for z_i in z:
            loss = loss + 1e-2 * torch.mean(z_i**2)
        
        if self.sep_loss:
            sep_loss = self.separation_loss(self.model.decoder)
            self.log(f'sep_loss/train', sep_loss, on_step=False, 
                                    on_epoch=True, prog_bar=True)
            
            loss = loss + sep_loss*self.sep_lr
            
        if self.zero_loss:
            z_zeros = [torch.zeros(1, self.hidden//self.num_encoders, z[0].shape[-1]).to(self.device) for _ in range(self.num_encoders)]
            x_pred_zeros = self.model.decode(z_zeros, True)
            zero_recon_loss = self.loss(x_pred_zeros, torch.zeros_like(x_pred_zeros))
            loss = loss + zero_recon_loss*self.zero_lr
            self.log(f'zero_recon_loss/train', zero_recon_loss, on_step=False, 
                                on_epoch=True, prog_bar=True)

        if batch_idx % 50 == 0:
            plot = plot_signal(x, torch.sigmoid(x_pred))
            self.logger.experiment.add_image('images/train_plot', plot, batch_idx)
        
        if batch_idx % self.plot_step == 0 and self.save_plots:
            self.save_inference_samples(self.x_plot)
            self.save_weight_visualizations()
        
        return loss
            
    def validation_step(self, batch, batch_idx):
        x, y = batch

        x_pred, z = self.model(x)
        recon_loss = self.loss(x_pred, x)
        self.log(f'recon_loss/val', recon_loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
                
        return recon_loss
    
    def _local_normalize(self, x):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        
        return x
    
    def save_inference_samples(self, patient):
        step = self.global_step
        
        save_name = os.path.join(self.plot_dir, f'images_{step}.png')
        x = [torch.tensor(x_i[0]).unsqueeze_(0).to(self.device) for x_i in patient]
        r = [torch.tensor(r_i[1]).unsqueeze_(0).to(self.device) for r_i in patient]
        x = torch.stack(x, dim=0).type(torch.float32).to(self.device)
        x = self._local_normalize(x)
        r = torch.stack(r, dim=0).type(torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x)
            zeros = torch.zeros_like(z[0])
            y = []
            for z_a in range(len(z)):
                z_pad = [zeros for _ in range(len(z))]
                z_pad[z_a] = z[z_a]
                y_i = self.model.decode(z_pad)
                y_i = torch.sigmoid_(y_i).squeeze_()
                y.append(y_i.cpu().numpy())

        title = f'Step: {step}'
                
        colors = ['#24e254', '#12e29f', '#00e1e9', '#05b5f4', '#0a89ff', '#535def', '#9c31df', '#c91976', '#f5000c', '#f5000c', '#f5000c', '#f5000c'] ##https://coolors.co/24e254-12e29f-00e1e9-05b5f4-0a89ff-535def-9c31df-c91976-f5000c
        clip = 512
        line_width = 1.5
        idx = 1
        matplotlib.rcParams['axes.linewidth'] = line_width
        matplotlib.rcParams['ytick.major.width'] = line_width
        matplotlib.rcParams['xtick.major.width'] = line_width
        default_c = '#434343'
        matplotlib.rcParams.update({'text.color' : f'{default_c}',
                                    'axes.labelcolor' : f'{default_c}',
                                    'axes.edgecolor' : f'{default_c}',
                                    'xtick.color' : f'{default_c}', 
                                    'ytick.color' : f'{default_c}'})

        plt.figure(figsize=(8,10))
        plt.title(title, pad=125)
        plt.tick_params(axis='both', direction='in')
        plt.grid(True, alpha=0.35, linewidth=line_width)
        plt.margins(x=0)
        
        # plot
        x_cpu_ = x[idx].detach().squeeze().cpu().numpy()
        x_cpu = z_norm(x_cpu_[clip:-clip])
        plt.plot(np.arange(len(x_cpu)), x_cpu, label=f'{self.input_signal_type} (Input)', c='#787878', linewidth=line_width)
        resp_ = r[idx].detach().squeeze().cpu().numpy()
        resp_ = z_norm(resp_[clip:-clip])
        plt.plot(np.arange(len(resp_)), resp_-8, label='Respiratory (Reference)', c='darkgray', linewidth=line_width)
        
        for i, y_i in enumerate(y):
            plt.plot(np.arange(len(y_i[idx][clip:-clip])), z_norm(y_i[idx][clip:-clip])-8*(i+2), label=f'Enc. {i} Source Pred.', alpha=0.9, c=colors[i+1], linewidth=line_width)
        
        plt.subplots_adjust(top=0.8, bottom=0.05, left=0.05, right=0.95)
        plt.yticks([])
        plt.legend(loc='lower center', frameon=False, fancybox=True, ncol=2, 
           bbox_to_anchor=(0.5, 1.015), fontsize='medium', markerfirst=False)
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
            if name.split('.')[-1] == 'weight' and len(w.shape) == 3:
                weights.append(torch.sum(torch.abs(w.detach()), dim=-1).cpu().numpy())
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
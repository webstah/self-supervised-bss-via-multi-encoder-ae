import io
import PIL.Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
from torch.nn.functional import sigmoid

from torchvision.transforms import ToTensor

def plot_grid(x, x_pred):
    x, x_pred = x.float().cpu().numpy(), sigmoid(x_pred.detach()).cpu().numpy()
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(2, 2),
                 axes_pad=0.1,
                 )
    im_list = [x[0], x_pred[0], x[1], x_pred[1]]
    
    for i in range(len(im_list)):
        im_list[i] = np.transpose(im_list[i], (1, 2, 0))
        
    for ax, im in zip(grid, im_list):
        im = im.astype(np.float32)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(im)

    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png')
    plot_buf.seek(0)

    image = PIL.Image.open(plot_buf)
    image = ToTensor()(image)
    plt.close()

    return image

def plot_signal(x, x_pred):
    x, x_pred = x.squeeze().cpu().numpy(), x_pred.squeeze().detach().cpu().numpy()
    x, x_pred = x[0], x_pred[0]
    # make plot
    plt.figure(figsize=(12,3))
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.title(f'Prediction and Target Comparison')
    plt.step(np.arange(0, x.shape[0]), x, linewidth=1.0, label='Target')
    plt.step(np.arange(0, x_pred.shape[0]), x_pred, linewidth=1.0, label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('Class/Amplitude')
    plt.legend()
    
    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png')
    plot_buf.seek(0)

    image = PIL.Image.open(plot_buf)
    image = ToTensor()(image)
    plt.close()

    return image

def z_norm(x):
    mean = np.percentile(x, 90)
    std = np.nanstd(x)

    return (x - mean) / (std + 1e-18)
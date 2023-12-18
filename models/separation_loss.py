import torch
import torch.nn as nn

class WeightSeparationLoss(nn.Module):
    def __init__(self, num_splits, mode='L1'):
        super(WeightSeparationLoss, self).__init__()
        self.num_splits = num_splits
        self.mode = mode
        assert mode in ['L1', 'L2'], f'Weight separation loss has an invalid argument for the \
            normalization mode: {mode}. Must be either L1 or L2.'

    def forward(self, model_item):
        loss = 0
        for name, w in model_item.named_parameters():
            if name.split('.')[-1] == 'weight' and len(w.shape) in [2, 3, 4]:
                w_out = w.shape[0]//self.num_splits
                w_in = w.shape[1]//self.num_splits
                for i in range(self.num_splits):
                    for j in range(self.num_splits):
                        if self.mode == 'L1' and i != j:
                            loss += torch.mean(torch.abs(w[w_out*i:w_out*(i+1), w_in*j:w_in*(j+1)]))
                        elif self.mode == 'L2' and i != j:
                            loss += torch.mean(torch.abs(w[w_out*i:w_out*(i+1), w_in*j:w_in*(j+1)]**2))
                    
        return loss

class WeightSeparationLossAlternative(nn.Module):
    def __init__(self, num_splits, mode='L1'):
        super(WeightSeparationLossAlternative, self).__init__()
        self.num_splits = num_splits
        self.mode = mode
        assert mode in ['L1', 'L2'], f'Weight separation loss has an invalid argument for the \
            normalization mode: {mode}. Must be either L1 or L2.'

    def forward(self, model_item):
        loss = 0
        for name, w in model_item.named_parameters():
            if name.split('.')[-1] == 'weight' and len(w.shape) in [2, 3, 4]:
                w_l = w.shape[0]//self.num_splits
                w_w = w.shape[1]//self.num_splits
                for i in range(self.num_splits-1):
                    if self.mode == 'L1':
                        loss += torch.mean(torch.abs(w[i*w_l:(i+1)*w_l, (i+1)*w_w:]))
                        loss += torch.mean(torch.abs(w[(i+1)*w_l:(i+2)*w_l, :(i+1)*w_w]))
                    elif self.mode == 'L2':
                        loss += torch.mean(w[i*w_l:(i+1)*w_l, (i+1)*w_w:]**2)
                        loss += torch.mean(w[(i+1)*w_l:(i+2)*w_l, :(i+1)*w_w]**2)
                    
        return loss

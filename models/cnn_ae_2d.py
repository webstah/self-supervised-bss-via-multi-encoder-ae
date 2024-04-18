import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, input_padding=0):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding+input_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, momentum=0.8),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels, momentum=0.8),
        )
        
    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_padding=0,
                 kernel_size=7, stride=1, padding=3, outlayer=False):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, stride=stride, 
                                padding=padding+input_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, momentum=0.8),
            nn.ConvTranspose2d(in_channels=out_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, stride=stride, 
                                padding=padding+input_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, momentum=0.8)
        )
        
    def forward(self, x):
        return self.block(x)

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, hidden=32, channels=[16, 32, 64]):
        super(ConvolutionalAutoencoder, self).__init__()
        self.input_channels = input_channels
        self.channels = [input_channels] + channels
        
        self.encoder = nn.Sequential()
        for i in range(0, len(self.channels)-1):
            self.encoder.append(EncoderBlock(self.channels[i], self.channels[i+1]))
            
        self.encoder_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channels[-1], 
                                out_channels=hidden, 
                                kernel_size=1, stride=1, 
                                padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden, momentum=0.8)
        )
        
        self.decoder_in = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden, 
                                out_channels=channels[-1], 
                                kernel_size=1, stride=1, 
                                padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[-1], momentum=0.8)
        )
        
        self.decoder = nn.Sequential()
        for i in reversed(range(1, len(self.channels)-1)):
            self.decoder.append(DecoderBlock(self.channels[i+1], self.channels[i]))
        self.decoder.append(DecoderBlock(self.channels[i], self.channels[i]))
        self.decoder_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channels[i], 
                                out_channels=output_channels, 
                                kernel_size=1, stride=1, 
                                padding=0),
        )
        
    def forward(self, x):
        # encode
        x = self.encoder(x)
        x = self.encoder_out(x)
        
        # decode
        y = self.decoder_in(x)
        y = self.decoder(y)
        
        # output
        y = self.decoder_out(y)
        
        return y
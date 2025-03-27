import torch
import torch.nn as nn
import torch.nn.functional as F

class Track_Model1(nn.Module):
    def __init__(self):
        super(Track_Model1, self).__init__()
        # Based on
        # Ronneberger, et al., “U-Net: Convolutional Networks for Biomedical Image Segmentation”, 2015

        self.down_layers = nn.ModuleList([
            self.conv_down_block(3, 64),
            self.conv_down_block(64, 128),
            self.conv_down_block(128, 256),
            self.conv_down_block(256, 512),
            self.conv_down_block(512, 1024),
            self.conv_down_block(1024, 2048),
        ])
        
        self.final_conv_layers = nn.ModuleList([
            nn.Conv2d(2048, 4096, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=0),
            nn.ReLU(),
        ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(12288, 256),
            nn.ReLU(),
            nn.Linear(256, 14),
        ])

    def conv_down_block(self, in_channels, out_channels):
        return nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

    def forward(self, x):
        for block in self.down_layers:
            for layer in block:
                #print(layer)
                x = layer(x)
                
        for layer in self.final_conv_layers:
            x = layer(x)
        
        x = torch.flatten(x,start_dim=1)
        
        for layer in self.linear_layers:
            x = layer(x)

        x = torch.reshape(x,(-1,2,7))
        
        return x


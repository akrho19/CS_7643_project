import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelSimple(nn.Module):
    def __init__(self):
        super(ModelSimple, self).__init__()
        
        self.layers = nn.ParameterList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
            nn.Sigmoid(),            
        ])

    def forward(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

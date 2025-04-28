import torch
import torch.nn as nn
import torch.nn.functional as F

# SUPER BASIC
class PoseModel(nn.Module):
    def __init__(self, out_channels):
        super(PoseModel, self).__init__()
        
        # Input: 64x64

        self.layers = nn.ModuleList([
            nn.Conv2d(5, 16, kernel_size=3), # 62
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), # 60
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), # 58
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 29
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear((64*29*29), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.Sigmoid(),
        ])


    def forward(self, x):
        # input_shape = x.shape

        for layer in self.layers:
            x = layer(x)


        return x

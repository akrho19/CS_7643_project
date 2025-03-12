import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelSimple(nn.Module):
    def __init__(self):
        super(ModelSimple, self).__init__()
        
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(64, 4, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        print(x.shape)
        # Encoder
        x = torch.relu(self.encoder_conv1(x))
        print(x.shape)
        x = self.pool(torch.relu(self.encoder_conv2(x)))
        print(x.shape)
        # Decoder
        x = self.upsample(torch.relu(self.decoder_conv1(x)))
        print(x.shape)
        x = self.decoder_conv2(x)
        print(x.shape)
        return x
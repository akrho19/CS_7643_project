import torch
import torch.nn as nn
import torch.nn.functional as F

# +

# SUPER BASIC
class Model1(nn.Module):
    def __init__(self, out_channels):
        super(Model1, self).__init__()
        # Based on
        # Ronneberger, et al., “U-Net: Convolutional Networks for Biomedical Image Segmentation”, 2015
        # Without the skip connections for now
        self.down_layers = nn.ModuleList([
            self.conv_down_block(3, 64),
            self.conv_down_block(64, 128),
            self.conv_down_block(128, 256),
            self.conv_down_block(256, 512),
        ])

        self.up_layers = nn.ModuleList([
            self.conv_up_block(512, 1024, 512),
            self.conv_up_block(1024, 512, 256),
            self.conv_up_block(512, 256, 128),
            self.conv_up_block(256, 128, 64),
        ])

        self.final_layers = nn.ModuleList([
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
            #nn.Softmax(dim=1),
        ])

    def conv_down_block(self, in_channels, out_channels):
        return nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
    
    def conv_up_block(self, in_channels, hidden_channels, out_channels):
        return nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=2, stride=2),
        ])

    def forward(self, x):
        # input_shape = x.shape

        x_skip = []
        for block in self.down_layers:
          for layer in block:
            if isinstance(layer, nn.Identity):
                x_skip.append(x)
                # print(x.shape)
            else:
                x = layer(x)

        
        # print("Finished down, starting up")
        i = 1
        for block in self.up_layers:
            # print(i)
            for layer in block:
                x = layer(x)
            # print(x.shape)
            # print(x_skip[-i].shape)

            x_skip_shape = x_skip[-i].shape

            crop_start_h = (x_skip_shape[2] - x.shape[2])//2
            crop_start_w = (x_skip_shape[3] - x.shape[3])//2
            # print(crop_start_h, crop_start_w)
            x_skip[-i] =  x_skip[-i][:,:,crop_start_h:(x.shape[2] + crop_start_h), \
                                        crop_start_w:(x.shape[3] + crop_start_w) ]
            

            x = torch.cat((x_skip[-i], x), dim=1)
            
            i += 1

        for layer in self.final_layers:
            x = layer(x)


        return x


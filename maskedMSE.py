# inspired by https://discuss.pytorch.org/t/how-to-write-a-loss-function-with-mask/53461

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, target, mask):
        B = inputs.shape[0]
        diff2 = (inputs - target) ** 2.0
        diff2[mask, 0:7] = 0.0
        result = torch.sum(diff2) / B
        return result

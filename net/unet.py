""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

unet.py - the U-Net version of our model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from . import model_parts as mp


def num_flat_features(x):
    ''' A utility function that returns the number of features in 
    the input image.'''
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class NetU(nn.Module):
    ''' U-Net code, similiar to the above. Taken from
    https://github.com/milesial/Pytorch-UNet/.'''

    def __init__(self, dtype=torch.float16):
        super(NetU, self).__init__()
        self.n_channels = 1
        self.n_classes = 5 # 5 probabilities for each of the neurons or background. TODO - might need to change this to 4?
        self.bilinear = True

        self.inc = mp.DoubleConv(1, 64, dtype=dtype)
        self.down1 = mp.Down(64, 128, dtype=dtype)
        self.down2 = mp.Down(128, 256, dtype=dtype)
        self.down3 = mp.Down(256, 512, dtype=dtype)
        self.down4 = mp.Down(512, 512, dtype=dtype)
        self.up1 = mp.Up(1024, 256, self.bilinear, dtype=dtype)
        self.up2 = mp.Up(512, 128, self.bilinear, dtype=dtype)
        self.up3 = mp.Up(256, 64, self.bilinear, dtype=dtype)
        self.up4 = mp.Up(128, 64, self.bilinear, dtype=dtype)
        self.outc = mp.OutConv(64, self.n_classes, dtype=dtype)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

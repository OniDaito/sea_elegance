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


class NetUBase(nn.Module):
    ''' U-Net code. Taken from
    https://github.com/milesial/Pytorch-UNet/.'''

    def __init__(self, dtype=torch.float32):
        super(NetUBase, self).__init__()
        self.n_channels = 1
        self.n_classes = 5
        self.trilinear = True
        self.inc = mp.DoubleConv(1, 64, dtype=dtype)
        self.down1 = mp.Down(64, 128, dtype=dtype)
        self.down2 = mp.Down(128, 256, dtype=dtype)
        self.down3 = mp.Down(256, 256, dtype=dtype)
        self.up1 = mp.Up(512, 256, trilinear=self.trilinear, dtype=dtype)
        self.up2 = mp.Up(384, 128, trilinear=self.trilinear, dtype=dtype)
        self.up3 = mp.Up(192, 64, trilinear=self.trilinear, dtype=dtype)
        self.outc = mp.OutConv(64, self.n_classes, dtype=dtype)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.outc(x)
        return out


class NetU(nn.Module):
    ''' U-Net code, Taken from
    https://github.com/milesial/Pytorch-UNet/.'''

    def __init__(self, dtype=torch.float32):
        super(NetU, self).__init__()
        self.n_channels = 1
        self.n_classes = 5
        self.trilinear = True
        self.inc = mp.DoubleConv(1, 32, dtype=dtype)
        self.down1 = mp.Down(32, 64, dtype=dtype)
        self.down2 = mp.Down(64, 128, dtype=dtype)
        self.down3 = mp.Down(128, 256, dtype=dtype)
        self.up1 = mp.Up(384, 128, trilinear=self.trilinear, dtype=dtype)
        self.up2 = mp.Up(192, 64, trilinear=self.trilinear, dtype=dtype)
        self.up3 = mp.Up(96, 32, trilinear=self.trilinear, dtype=dtype)
        self.outc = mp.OutConv(32, self.n_classes, dtype=dtype)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.outc(x)
        return out
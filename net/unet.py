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
import math
import random
import argparse
from model_parts import Down, DoubleConv, OutConv

def num_flat_features(x):
  ''' A utility function that returns the number of features in 
  the input image.'''
  size = x.size()[1:]  # all dimensions except the batch dimension
  num_features = 1
  for s in size:
    num_features *= s
  return num_features

class NetEncDec(nn.Module):
    """ Our Encoder decoder network that takes in a batch of 
    images of SIZE, shrinks down, then expands to our mask
    style layer. """
    def __init__(self) :
        super(NetEncDec, self).__init__()
        # Batch norm layers
        self.batch1 = nn.BatchNorm2d(16)
        self.batch2 = nn.BatchNorm2d(32)
        self.batch3 = nn.BatchNorm2d(64)
        self.batch4 = nn.BatchNorm2d(128)
        self.batch5 = nn.BatchNorm2d(256)
        self.batch6 = nn.BatchNorm2d(256)

        # Conv layers
        #Added more conf layers as we aren't using maxpooling 
        self.conv1 = nn.Conv2d(1, 16, 5, stride = 2, padding = 1 )
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 1 )
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 1 )
        self.conv4 = nn.Conv2d(64, 128, 5, stride = 2, padding = 1 ) 
        self.conv5 = nn.Conv2d(128, 256, 5, stride = 2, padding = 1 ) 
        self.conv6 = nn.Conv2d(256, 256, 5, stride = 2, padding = 1 )
        self.fc1 = nn.Linear(16384, 512) # seems like a lot :/
    
        self.deconv1 = nn.ConvTranspose2d(256, 256, 5, stride = 2, padding = 1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, stride = 2, padding = 1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, stride = 2, padding = 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 5, stride = 2, padding = 1)
        self.deconv5 = nn.ConvTranspose2d(32, 16, 5, stride = 2, padding = 1)
        self.deconv6 = nn.ConvTranspose2d(16, 1, 5, stride = 2, padding = 1,\
            output_padding = 1)

        self.device = "cpu"
        
    def to(self, device):
        super(NetEncDec, self).to(device)
        self.device = device
        return self

    def forward(self, target):
        """ Take in our images of batch_size x width x height
        then shrink down then expand."""    
        x = F.leaky_relu(self.batch1(self.conv1(target)))
        x = F.leaky_relu(self.batch2(self.conv2(x)))
        x = F.leaky_relu(self.batch3(self.conv3(x)))
        x = F.leaky_relu(self.batch4(self.conv4(x)))
        x = F.leaky_relu(self.batch5(self.conv5(x)))
        x = F.leaky_relu(self.batch6(self.conv6(x)))

        #ff = num_flat_features(x)
        #x = x.view(-1, ff)
        #x = F.leaky_relu(self.fc1(x))
        #x = x.view(-1, 128, 2, 2)

        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = F.leaky_relu(self.deconv3(x))
        x = F.leaky_relu(self.deconv4(x))
        x = F.leaky_relu(self.deconv5(x))
        x = F.leaky_relu(self.deconv6(x))
    
        return x

class NetU(nn.Module):
    ''' U-Net code, similiar to the above. Taken from
    https://github.com/milesial/Pytorch-UNet/.'''

    def __init__(self):
        super(NetU, self).__init__()
        self.n_channels = 1
        self.n_classes = 1
        self.bilinear = True

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, self.bilinear)
        self.up2 = Up(512, 128, self.bilinear)
        self.up3 = Up(256, 64, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

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

""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

model_parts.py - small classes used inside the u-net
module

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dtype=dtype),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=dtype),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool3d(2),
            nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2, dtype=dtype),
            DoubleConv(in_channels, out_channels, dtype=dtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, dtype=torch.float32):
        super().__init__()

        # Potentially replace upsample
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.upsample.html?highlight=upsample#torch.nn.functional.upsample
        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, dtype=dtype)

        self.conv = DoubleConv(in_channels, out_channels, dtype=dtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [
                        diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, dtype=dtype)

    def forward(self, x):
        return self.conv(x)

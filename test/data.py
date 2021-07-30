""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

data.py - our test functions using python's unittest.
This file tests the data classes like loader and such.

"""
import unittest
import math
import torch
import random
from data.loader import WormDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage as nd



class Data(unittest.TestCase):
    def test_loader(self):
        worm_data = WormDataset(annotations_file="./test/images/test_data.csv", img_dir='./test/images')
        #train_size = 6
        #test_size = 3
        #train_dataset, test_dataset = torch.utils.data.random_split(worm_data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

        dataloader = DataLoader(worm_data, batch_size=3, shuffle=False)
        #test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False)

        train_source, train_asi, train_asj = next(iter(dataloader))
        print(f"Feature batch shape: {train_source.size()}")
        print(f"Labels batch shape: {train_source.size()}")
        print("Data type", train_source.dtype)
     
        #print("value", train_source[0][0][114][517])
        self.assertTrue(train_asi[0][0][121][501] == 1)
        self.assertTrue(train_source[0][0][112][512] == 1467)
        self.assertTrue(train_source[0][0][114][517] == 1612)

        plt.figure()
        f, axarr = plt.subplots(3, 1)
        axarr[0].imshow(train_source[0].squeeze().numpy())
        axarr[1].imshow(train_asi[0].squeeze().numpy())
        axarr[2].imshow(train_asj[0].squeeze().numpy())
        plt.show()

    def test_scale(self):
        img_path = "./test/images/raw.fits"
        with fits.open(img_path) as w:
            hdul = w[0].data.byteswap().newbyteorder()
            source_image = np.array(hdul).astype("int16")
            source_image = nd.interpolation.zoom(source_image, zoom=0.5)
            c = plt.imshow(source_image[13])
            plt.show()

    def test_reduce(self):
        img_path = "./test/images/asi.fits"
        with fits.open(img_path) as w:
            hdul = w[0].data.byteswap().newbyteorder()
            source_image = np.array(hdul).astype("int8")
            reduced = np.max(source_image.astype(float), axis=1)
            c = plt.imshow(reduced)
            plt.show()
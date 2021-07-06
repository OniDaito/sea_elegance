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


class Data(unittest.TestCase):
    def test_loader(self):
        worm_data = WormDataset(annotations_file="./test/images/test_data.csv", img_dir='./test/images')

        train_size = int(0.6 * len(worm_data))
        test_size = len(worm_data) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(worm_data, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=True)

        train_features, train_labels = next(iter(train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")


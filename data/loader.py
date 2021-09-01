""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

loader.py - loaders, datasets and the like for all our images

"""

import os
import torch
import pandas as pd
import numpy as np
from torch._C import device
from torch.utils.data import Dataset
from PIL import Image
from astropy.io import fits
from scipy import ndimage as nd


def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


def binaryise(input_data: np.ndarray) -> np.ndarray:
    ''' Convert the tensors so we don't have different numbers. If its
    not a zero, it's a 1.'''
    res = input_data.copy()
    res[input_data != 0] = 1
    return res


def make_sparse(input_data: np.ndarray, dtype, device):
    indices = []
    data = []
    
    for b in range(input_data.shape[0]):
        for z in range(input_data.shape[1]):
            for y in range(input_data.shape[2]):
                for x in range(input_data.shape[3]):
                    if input_data[b][z][y][x] != 0:
                        indices.append([b, z, y, x])
                        data.append(input_data[b][z][y][x])

    print("SHAPES", len(indices), len(data))
    s = torch.sparse_coo_tensor(list(zip(*indices)), data, torch.Size(input_data.shape))
    s.to(dtype)
    s.to(device)
    return s


class WormDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, device='cpu'):
        self.img_targets = pd.read_csv(annotations_file, names=["source", "asi", "asj"],
                    converters = {'source' : strip,
                                    'asi' : strip,
                                    'asj' : strip})
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.img_targets)

    def __getitem__(self, idx):
        # Using PIL here and making sure we set uint16 as PIL incorrecly reads this as uint8
        # torchvision was in the original code example but it fails to read 16bit pngs
        # It looks like torch doesn't do 16 bit unsigned, so lets convert and hope it's ok
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 0])

        with fits.open(img_path) as w:
            hdul = w[0].data.byteswap().newbyteorder()
            source_image = np.array(hdul).astype("int16")
            source_image = nd.interpolation.zoom(source_image, zoom=0.5)
            source_image = source_image.astype(float) / 4095.0
            source_image = source_image.astype(np.float32)
            source_image = np.expand_dims(source_image, axis=0)
            # Divide by the maximum possible in order to normalise the input. Should help with
            # exploding gradients and optimisation.
            source_image = torch.tensor(source_image, dtype=torch.float32, device=self.device)

        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 1])

        with fits.open(img_path) as w:
            hdul = w[0].data.byteswap().newbyteorder()
            target_asi = np.array(hdul).astype("int8")
            target_asi = nd.interpolation.zoom(target_asi, zoom=0.5)
            target_asi = binaryise(target_asi)
            target_asi = np.expand_dims(target_asi, axis=0)
            target_asi = make_sparse(target_asi, torch.int8, self.device)
   
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 2])
        
        with fits.open(img_path) as w:
            hdul = w[0].data.byteswap().newbyteorder()
            target_asj = np.array(hdul).astype("int8")
            target_asj = nd.interpolation.zoom(target_asj, zoom=0.5)
            target_asj = binaryise(target_asj)
            target_asj = np.expand_dims(target_asj, axis=0)
            target_asj = make_sparse(target_asj, torch.int8, self.device)

        if self.transform:
            source_image = self.transform(source_image)

        if self.target_transform:
            target_asi = self.target_transform(target_asi)
            target_asj = self.target_transform(target_asj)

        return source_image, target_asi, target_asj

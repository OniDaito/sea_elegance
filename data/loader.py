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


def make_sparse(input_data: np.ndarray, device):
    indices = []
    data = []
    
    for z in range(input_data.shape[0]):
        for y in range(input_data.shape[1]):
            for x in range(input_data.shape[2]):
                if input_data[z][y][x] != 0:
                    indices.append([z, y, x])
                    data.append(input_data[z][y][x])

    s = torch.sparse_coo_tensor(list(zip(*indices)), data, torch.Size(input_data.shape))
    s.to(device)
    return s


class WormDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, device='cpu'):
        super().__init__()
        self.img_targets = pd.read_csv(annotations_file, names=["source", "mask"],
                    converters = {'source' : strip,
                                    'mask' : strip})
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

        # TODO - should normalised range go from -1 to 1?

        with fits.open(img_path) as w:
            hdul = w[0].data.byteswap().newbyteorder()
            source_image = np.array(hdul).astype("int16")
            #source_image = nd.interpolation.zoom(source_image, zoom=0.5)
            source_image = source_image.astype(float) / 4095.0
            source_image = source_image.astype(np.float32)
            source_image = np.expand_dims(source_image, axis=0)
            # Divide by the maximum possible in order to normalise the input. Should help with
            # exploding gradients and optimisation.
            source_image_final = torch.tensor(source_image, dtype=torch.float32, device=self.device)
        
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 1])

        with fits.open(img_path) as w:
            # We can keep the masks as ordinals and not split into a number of dimensions
            hdul = w[0].data.byteswap().newbyteorder()
            target_mask = np.array(hdul).astype("int8")
            # nd interpolation will do 3D BUT it invents new classes (-1 and 5) which messes
            # everything up, so we add a clamp. This might not be ideal and could break
            # things in the future, so we'll need to add a 3D resize in the dataset creation
            # in the wiggle project.
            #target_mask = nd.interpolation.zoom(target_mask, zoom=0.5)
            #target_mask = np.clip(target_mask, 0, 4)

            assert(np.all(target_mask >= 0))
            assert(np.all(target_mask < 5))
            target_mask = target_mask.astype(np.float32)
            # target_mask_final = make_sparse(target_mask, self.device)
            target_mask_final = torch.tensor(target_mask, dtype=torch.float32, device=self.device)
   
        if self.transform:
            source_image_final = self.transform(source_image_final)

        if self.target_transform:
            target_mask_final = self.target_transform(target_mask_final)

        return source_image_final, target_mask_final

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
from torch.utils.data import Dataset
from PIL import Image


class WormDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_targets = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_targets)

    def __getitem__(self, idx):
        # Using PIL here and making sure we set uint16 as PIL incorrecly reads this as uint8
        # torchvision was in the original code example but it fails to read 16bit pngs
        # It looks like torch doesn't do 16 bit unsigned, so lets convert and hope it's ok
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 0])
        source_image = torch.tensor(np.array(Image.open(img_path)).astype("uint16"), dtype=torch.short)

        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 1])
        target_asi = torch.tensor(np.array(Image.open(img_path)).astype("uint16"), dtype=torch.short)
   
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 2])
        target_asj = torch.tensor(np.array(Image.open(img_path)).astype("uint16"), dtype=torch.short)

        if self.transform:
            source_image = self.transform(source_image)

        if self.target_transform:
            target_asi = self.target_transform(target_asi)
            target_asj = self.target_transform(target_asj)

        return source_image, target_asi, target_asj

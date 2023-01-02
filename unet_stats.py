
""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

stats.py - look at the worm data and generate some stats

This script creates two files, an HDF5 and a CSV. The HDF5
file can be quite large - in the order of tens of gigabytes
depending on the size of the test set used with the U-Net.

Example use:
python unet_stats.py --base /phd/wormz/queelim --rep /media/proto_backup/wormz/queelim --dataset /media/proto_backup/wormz/queelim/dataset_3d_basic_noresize --savedir /media/proto_working/runs/wormz_2022_09_19 --no-cuda
python unet_stats.py --load summary_stats.h5

"""

import pickle
from turtle import back
import torch
import numpy as np
import argparse
import csv
import os
import h5py
import scipy
import torch.nn as nn
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from util.loadsave import load_model, load_checkpoint
from util.image import load_fits, save_fits
import torch.nn.functional as F
from tqdm import tqdm

data_files = [ 
["/phd/wormz/queelim/ins-6-mCherry/20170724-QL285_S1-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170724-QL285_S1-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170724-QL604_S1-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170724-QL604_S1-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170724-QL922_S1-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170724-QL922_S1-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170724-QL923_S1-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170724-QL923_S1-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170724-QL925_S1-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170724-QL925_S1-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170803-QL285_SB1-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170803-QL285_SB1-d1.0"], 
["/phd/wormz/queelim/ins-6-mCherry/20170803-QL285_SB2-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170803-QL285_SB2-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170803-QL604_SB2-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170803-QL604_SB2-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170803-QL922_SB2-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170803-QL922_SB2-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170803-QL923_SB2-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation//20170803-QL923_SB2-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170803-QL925_SB2-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170803-QL925_SB2-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170804-QL285_SB3-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170804-QL285_SB3-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170804-QL604_SB3-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170804-QL604_SB3-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170804-QL922_SB3-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170804-QL922_SB3-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170804-QL923_SB3-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170804-QL923_SB3-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry/20170804-QL925_SB3-d1.0", "/phd/wormz/queelim/ins-6-mCherry/Annotation/20170804-QL925_SB3-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170810-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170810-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170810-QL603-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170810-QL603-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170810-QL806-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170810-QL806-d1.0"], 
["/phd/wormz/queelim/ins-6-mCherry_2/20170810-QL867-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170810-QL867-d1.0"], 
["/phd/wormz/queelim/ins-6-mCherry_2/20170817-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170817-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170817-QL603-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170817-QL603-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170817-QL806-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170817-QL806-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170817-QL867-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170817-QL867-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170818-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170818-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170818-QL603-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170818-QL603-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170818-QL806-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170818-QL806-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170818-QL867-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Hai_analysis/20170818-QL867-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170821-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170821-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170821-QL569-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170821-QL569-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170821-QL849-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170821-QL849-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170825-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170825-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170825-QL569-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170825-QL569-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170825-QL849-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170825-QL849-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170828-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170828-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170828-QL569-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170828-QL569-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170828-QL849-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20170828-QL849-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170907-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170907-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170907-QL568-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170907-QL568-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170907-QL823-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170907-QL823-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170907-QL824-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170907-QL824-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170907-QL835-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170907-QL835-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170908-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170908-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170908-QL568-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170908-QL568-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170908-QL823-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170908-QL823-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170908-QL824-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170908-QL824-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170908-QL835-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170908-QL835-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170911-QL285-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170911-QL285-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170911-QL568-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170911-QL568-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170911-QL823-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170911-QL823-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170911-QL824-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170911-QL824-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20170911-QL835-d1.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/LH ARZ analysis/20170911-QL835-d1.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180126-QL285-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180126-QL285-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180126-QL569-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180126-QL569-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180126-QL849-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180126-QL849-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180201-QL285-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180201-QL285-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180201-QL569-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180201-QL569-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180201-QL849-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180201-QL849-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180202-QL285-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180202-QL285-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180202-QL417-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180202-QL417-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180202-QL787-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180202-QL787-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180202-QL795-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180202-QL795-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180208-QL285-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180208-QL285-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180302-QL285-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180302-QL285-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180302-QL285-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180302-QL285-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180302-QL849-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180302-QL849-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180308-QL285-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180308-QL285-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180308-QL569-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180308-QL569-d0.0"],
["/phd/wormz/queelim/ins-6-mCherry_2/20180308-QL849-d0.0", "/phd/wormz/queelim/ins-6-mCherry_2/Annotations/Reesha_analysis/20180308-QL849-d0.0"]
]

# TODO - remove from globals and have on command line (or a check first off)
# For now, set these manually
image_depth = 51
image_height = 200
image_width = 200
nclasses = 3
chunk_size = 2

def find_background(og_image):
    '''
    kernel_dia = 3
    kernel_rad = 1
    modes = []

    for z in range (kernel_rad, og_image.shape[0] - kernel_rad):
        for y in range (kernel_rad, og_image.shape[1] - kernel_rad):
            for x in range (kernel_rad, og_image.shape[2] - kernel_rad):

                # Add up all the pixels - 27 of them
                avg = og_image[z][y][x]
                # First 9
                avg += og_image[z-kernel_rad][y-kernel_rad][x-kernel_rad]
                avg += og_image[z-kernel_rad][y-kernel_rad][x]
                avg += og_image[z-kernel_rad][y-kernel_rad][x+kernel_rad]
                avg += og_image[z-kernel_rad][y][x-kernel_rad]
                avg += og_image[z-kernel_rad][y][x]
                avg += og_image[z-kernel_rad][y][x+kernel_rad]
                avg += og_image[z-kernel_rad][y+kernel_rad][x-kernel_rad]
                avg += og_image[z-kernel_rad][y+kernel_rad][x]
                avg += og_image[z-kernel_rad][y+kernel_rad][x+kernel_rad]

                # Mid 8
                avg += og_image[z][y-kernel_rad][x-kernel_rad]
                avg += og_image[z][y-kernel_rad][x]
                avg += og_image[z][y-kernel_rad][x+kernel_rad]
                avg += og_image[z][y][x-kernel_rad]
                avg += og_image[z][y][x+kernel_rad]
                avg += og_image[z][y+kernel_rad][x-kernel_rad]
                avg += og_image[z][y+kernel_rad][x]
                avg += og_image[z][y+kernel_rad][x+kernel_rad]

                # Last 9
                avg += og_image[z+kernel_rad][y-kernel_rad][x-kernel_rad]
                avg += og_image[z+kernel_rad][y-kernel_rad][x]
                avg += og_image[z+kernel_rad][y-kernel_rad][x+kernel_rad]
                avg += og_image[z+kernel_rad][y][x-kernel_rad]
                avg += og_image[z+kernel_rad][y][x]
                avg += og_image[z+kernel_rad][y][x+kernel_rad]
                avg += og_image[z+kernel_rad][y+kernel_rad][x-kernel_rad]
                avg += og_image[z+kernel_rad][y+kernel_rad][x]
                avg += og_image[z+kernel_rad][y+kernel_rad][x+kernel_rad]

                avg /= 27.0
                modes.append(int(avg))
    '''

    from scipy import signal

    kernel = np.array([1.0 / 27.0 for i in range(27)]).reshape((3,3,3))
    filtered = signal.convolve(og_image, kernel, mode='same')
    modes = filtered.astype(int)
        
    vals, counts = np.unique(modes, return_counts=True)
    mode_value = np.argwhere(counts == np.max(counts))
    return vals[mode_value[0][0]]


def find_border(image):
    ''' Find the border of an image where the background is 0 and the actual area is 1.'''
    shifted_x = np.roll(image, -1, axis=2)
    shifted_y = np.roll(image, -1, axis=1)
    shifted_z = np.roll(image, -1, axis=0)
 
    #print(shifted)
    border_x = np.abs(image - shifted_x)
    border_y = np.abs(image - shifted_y)
    border_z = np.abs(image - shifted_z)
    border = border_x + border_y + border_z
    border = np.clip(border, 0, 1)
    border = np.reshape(border, image.shape)
    return border
    

def find_image_pairs(args):
    ''' Find all the images we need and pair them up properly with correct paths.
    We want to find the original fits images that correspond to our test dataset.'''
    idx_to_mask = {}
    idx_to_source = {}
    prefix_to_roi = {}
    idx_to_prefix = {}
    dataset = []
    file_lines = []

    # Read the dataset.log file - it should have the conversions
    if os.path.exists(args.dataset + "/dataset.log"):
        with open(args.dataset + "/dataset.log") as f:

            for line in f.readlines():
                file_lines.append(line)

    for lidx, line in enumerate(file_lines):
        prefix = ""

        if "Stacking" in line:
            source_tiff = line.replace("Stacking: ", "").replace("\n", "")

            for line2 in file_lines[lidx:]:

                if "Renaming" in line2 and "AutoStack" in line2 and source_tiff in line2:
                    x = 0
                    tokens = line2.split(" ")
                    mask = tokens[-1].replace("\n", "")

                    if len(tokens) != 4:
                        tokens = line2.split(" to ")
                        mask = tokens[1].replace("\n","")

                    for i in range(len(mask)):
                        if mask[i] == "/":
                            x = i

                    idx = int(mask[x+1:].replace("_layered.fits", ""))
                    dataset.append(idx)

                if "Pairing" in line2 and "AutoStack" and "WS2" in line2 and source_tiff in line2:
                    tokens = line2.split(" ")
                    mask = tokens[1]
                    source = tokens[-1].replace("\n","")

                    if len(tokens) != 6:
                        tokens = line2.split(" and ")
                        source = tokens[1].replace("\n","")
                        tokens = line2.split(" with ")
                        mask = tokens[0].replace("Pairing ", "")

                    x = 0
                    
                    for i in range(len(source)):
                        if source[i] == "/":
                            x = i
                    
                    # We want to find the FITS version of the original input image.
                    # TODO - maybe add this to the sources at the top of the script
                    source = source.replace(".tiff", ".fits").replace("ins-6-mCherry_2", "mcherry_2_fits").replace("ins-6-mCherry", "mcherry_fits")
                    idx_to_source[idx] = source
                    idx_to_mask[idx] = mask
            
        elif "ROI" in line:
            tokens = line.split(",")
            filepath = tokens[0]
            head, tail = os.path.split(filepath)
            head, pref = os.path.split(head)
            _, pref2 = os.path.split(head)
            final = pref2 + "/" + pref + "/" + tail
            final = final.replace("tiff", "")
            final = final.replace("_WS2","")
        
            roi = {}
            roi["xs"] = int(tokens[2])
            roi["ys"] = int(tokens[3])
            roi["zs"] = int(tokens[4])
            roi["xe"] = roi["xs"] + int(tokens[5])
            roi["ye"] = roi["ys"] + int(tokens[5])
            roi["ze"] = roi["zs"] + int(tokens[6])
            prefix_to_roi[final] = roi

    for idx in dataset:
        # print(idx)
        path = idx_to_mask[idx]
        head, tail = os.path.split(path)
        head, pref = os.path.split(head)
        _, pref2 = os.path.split(head)
        final = pref2 + "/" + pref + "/" + tail
        final = final.replace("tiff", "")
        final = final.replace("_WS2","")
        idx_to_prefix[idx] = final

    # Find the test set for a particular run
    sources_masks = []
    og_sources = []
    og_masks = []
    rois = []

    if os.path.exists(args.savedir + "/dataset_test.csv"):
        with open(args.savedir + "/dataset_test.csv") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            
            for row in reader:
                idx = int(row['source'].split("_")[0])
                path_source = args.dataset + "/" + row['source']
                path_mask = args.dataset + "/" + row[' target'].replace(" ", "")
                sources_masks.append((path_source, path_mask))
                og_sources.append(idx_to_source[idx])
                og_masks.append(idx_to_mask[idx])
                rois.append(prefix_to_roi[idx_to_prefix[idx]])
    

    print("Dataset size:", len(og_sources))
    return sources_masks, og_sources, og_masks, rois

def crop_image(image, roi, target_size):
    ''' Crop the image. However the crop might be bigger than the actual
    target size due to augmentation.'''
    xs = roi['xs']
    xe = roi['xe']
    ys = image.shape[1] - roi['ys']
    ye = image.shape[1] - roi['ye']
    width = xe - xs
    height = ye - ys

    if target_size[0] != width:
        dw = int((width - target_size[0]) / 2)
        xs = xs + dw
        xe = xe - dw

    if target_size[1] != height:
        dh = int((height - target_size[1]) / 2)
        ys = ys + dh
        ye = ye - dh

    assert(xe - xs == target_size[0])
    assert(ye - ys == target_size[1])

    cropped = image[:, ys:ye, xs:xe ]
    return cropped

def read_counts(args, sources_masks, og_sources, og_masks, rois):
    ''' For the test set, integrate the brightness under the masks
    from the OG sources, then predict the mask using the network and
    integrate that brightness too.'''


    # Now load the model to test it's predictions
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.savedir and os.path.isfile(args.savedir + "/checkpoint.pth.tar"):

        model = load_model(args.savedir + "/model.tar")
        (model, _, _, _, _, prev_args, _) = load_checkpoint(
            model, args.savedir, "checkpoint.pth.tar", device
        )
        model = model.to(device)
        model.eval()

        with h5py.File(args.save + ".h5", 'w') as hf:
            asi_actual_hf = hf.create_dataset("asi_actual", (1, image_depth, image_height, image_width), maxshape=(None,  image_depth, image_height, image_width), chunks=(chunk_size, image_depth, image_height, image_width))
            asj_actual_hf = hf.create_dataset("asj_actual", (1, image_depth, image_height, image_width), maxshape=(None,  image_depth, image_height, image_width), chunks=(chunk_size, image_depth, image_height, image_width))
            asi_pred_hf = hf.create_dataset("asi_pred", (1, image_depth, image_height, image_width), maxshape=(None,  image_depth, image_height, image_width), chunks=(chunk_size, image_depth, image_height, image_width))
            asj_pred_hf = hf.create_dataset("asj_pred", (1, image_depth, image_height, image_width), maxshape=(None,  image_depth, image_height, image_width), chunks=(chunk_size, image_depth, image_height, image_width))
            og_hf = hf.create_dataset("og", (1, image_depth, image_height, image_width), maxshape=(None,  image_depth, image_height, image_width), chunks=(chunk_size, image_depth, image_height, image_width))
            og_back_hf = hf.create_dataset("og_back", (1, image_depth, image_height, image_width), maxshape=(None,  image_depth, image_height, image_width), chunks=(chunk_size, image_depth, image_height, image_width))
            back_hf = hf.create_dataset("back", (1, 1), maxshape=(None, 1))

        with open(args.save + ".csv", "w") as w:
            w.write("idx,src,mask,Overlay,Overlap,Dice,Jaccard,Dice1,Jacc1,tp1,tn1,fp1,fn1,Dice2,Jacc2,tp2,tn2,fp2,fn2\n")

        with h5py.File(args.save + ".h5", 'a') as hf:
            asi_actual_hf = hf['asi_actual']
            asj_actual_hf = hf['asj_actual']
            asi_pred_hf = hf['asi_pred']
            asj_pred_hf = hf['asj_pred']
            og_hf = hf['og']
            og_back_hf = hf['og_back']
            back_hf = hf["back"]

    
            for fidx, paths in enumerate(sources_masks):
                print("Testing", paths)

                # When loading images, we may need to use a base 
                og_path = og_sources[fidx]
                source_path, target_path = paths

                if args.base != "" and args.rep != ".":
                    og_path = og_path.replace(args.base, args.rep)
                    source_path = source_path.replace(args.base, args.rep)
                    target_path = target_path.replace(args.base, args.rep)

                og_image = load_fits(og_path, dtype=torch.float32)

                background_value = find_background(og_image)
                print("Background value:", background_value)

                roi = rois[fidx]
            
                input_image = load_fits(source_path, dtype=torch.float32)
                target_image = load_fits(target_path, dtype=torch.float32)

                width = input_image.shape[-1]
                height = input_image.shape[-2]
                depth = input_image.shape[-3]
                og_image = crop_image(og_image, roi, (width, height, depth))

                # TODO - no resizing this time around
                resized_image = input_image # resize_3d(input_image, 0.5)
                normalised_image = resized_image / 4095.0

                with torch.no_grad():
                    im = normalised_image.unsqueeze(dim=0).unsqueeze(dim=0)
                    im = im.to(device)
                    prediction = model.forward(im)

                    #with open('prediction.npy', 'wb') as f:
                    #   np.save(f, prediction.detach().cpu().numpy())
                    assert(not(torch.all(prediction == 0).item()))

                    classes = prediction[0].detach().cpu().squeeze()
                    classes = F.one_hot(classes.argmax(dim=0), nclasses).permute(3, 0, 1, 2)
                    classes = np.argmax(classes, axis=0)
                    
                    # Why is this not working? Seems to stall
                    #resized_prediction = prediction # resize_3d(prediction, 2.0)
                    resized_prediction = classes
                    pred_filename = os.path.basename(source_path).replace("layered","pred")
                    save_fits(resized_prediction, pred_filename)
                    mask_filename = os.path.basename(source_path).replace("layered","mask")
                    save_fits(target_image, mask_filename)
                    source_filename = os.path.basename(source_path).replace("layered","source")
                    save_fits(og_image, source_filename)

                    print("Source Image", og_path)

                    og_image =  np.expand_dims(og_image, axis=0)
                    og_hf[-og_image.shape[0]:] = og_image
                    
                    if fidx + 1 < len(sources_masks):
                        og_hf.resize(og_hf.shape[0] + og_image.shape[0], axis = 0)

                    back_hf[-1:] = np.array([background_value])
                    
                    if fidx + 1 < len(sources_masks):
                        back_hf.resize(back_hf.shape[0] + 1, axis = 0)

                    #input_image = torch.narrow(input_image, 0, 0, resized_prediction.shape[0])
                    print("Shapes", classes.shape, input_image.shape)

                    asi_pred_mask = torch.where(resized_prediction == 1, 1, 0)
                    asj_pred_mask = torch.where(resized_prediction == 2, 1, 0)

                    asi_actual_mask = torch.where(target_image == 1, 1, 0)
                    asj_actual_mask = torch.where(target_image == 2, 1, 0)

                    og_image_back = og_image - background_value
                    og_image_back = np.clip(og_image_back, 0, 4096)

                    # Append the results of real masks to og backgroimages
                    og_back_hf[-og_image.shape[0]:] = og_image_back
                    
                    if fidx + 1 < len(sources_masks):
                        og_back_hf.resize(og_back_hf.shape[0] + og_image_back.shape[0], axis = 0)

                    print("Shape:", asi_actual_hf.shape)

                    # Append the results of real masks to og images
                    asi_actual_mask =  np.expand_dims(asi_actual_mask, axis=0)
                    asi_actual_hf[-asi_actual_mask.shape[0]:] = asi_actual_mask
                    
                    if fidx + 1 < len(sources_masks):
                        asi_actual_hf.resize(asi_actual_hf.shape[0] + asi_actual_mask.shape[0], axis = 0)

                    asj_actual_mask =  np.expand_dims(asj_actual_mask, axis=0)
                    asj_actual_hf[-asj_actual_mask.shape[0]:] = asj_actual_mask
                    
                    if fidx + 1 < len(sources_masks):
                        asj_actual_hf.resize(asj_actual_hf.shape[0] + asj_actual_mask.shape[0], axis = 0)

                    # Now append the predictions
                    asi_pred_mask =  np.expand_dims(asi_pred_mask, axis=0)
                    asi_pred_hf[-asi_pred_mask.shape[0]:] = asi_pred_mask
                    
                    if fidx + 1 < len(sources_masks):
                        asi_pred_hf.resize(asi_pred_hf.shape[0] + asi_pred_mask.shape[0], axis = 0)

                    asj_pred_mask =  np.expand_dims(asj_pred_mask, axis=0)
                    asj_pred_hf[-asj_pred_mask.shape[0]:] = asj_pred_mask
                
                    if fidx + 1 < len(sources_masks):
                        asj_pred_hf.resize(asj_pred_hf.shape[0] + asj_pred_mask.shape[0], axis = 0)

                    # Compute the Jaccard Scores

                    scores = compare_masks(target_image.numpy(), resized_prediction.numpy())

                    with open(args.save + ".csv", "a") as w:
                        w.write(str(fidx) + "," + source_path + "," + target_path + "," + scores + "\n")


def compare_masks(original: np.ndarray, predicted: np.ndarray):
    ''' Generate the Jaccard scores for the test set.'''
    
    # L1 Sum absolute differences
    l1 = np.subtract(original, predicted)
    l1 = np.absolute(l1)
    l1 = np.sum(l1)

    # Mask overlay score
    tsize = np.shape(original)
    usize = tsize[0] * tsize[1] * tsize[2]
    overlay = np.sum(np.where(original == predicted, 1, 0)) / usize
    score = str(overlay) + ","

    # Of the areas, how did we get it right?
    m = np.where(original != 0, 1, 0)
    n = np.where(predicted != 0, 1, 0)
    overlap = np.sum(m * n) / np.sum(m)
    score += str(overlap) + ","

    # Dice score
    dice = (2 * np.sum(m * n)) / (np.sum(m) + np.sum(n))
    score += str(dice) + ","

    # Jaccard
    jacc = np.sum(n * m) / (np.sum(n) + np.sum(m) - np.sum(n * m))
    score += str(jacc) + ","

    # Per class Dice and Jaccard
    fm = np.where(original == 0, 1, 0)
    fn = np.where(predicted == 0, 1, 0)

    for c in range(1, 3):
        m = np.where(original == c, 1, 0)
        n = np.where(predicted == c, 1, 0)
      
        dice = (2 * np.sum(m * n)) / (np.sum(m) + np.sum(n))
        jacc = np.sum(n * m) / (np.sum(n) + np.sum(m) - np.sum(n * m))
        score += str(dice) + ","
        score += str(jacc) + ","
        true_pos = np.sum(m * n)
        true_neg = np.sum(fm * fn)
        false_pos = np.sum(n * fm)
        false_neg = np.sum(fn * m)
        score += str(true_pos) + ","
        score += str(true_neg) + ","
        score += str(false_pos) + ","
        score += str(false_neg) + ","

    score = score[:-1]
    return score


def read_chunks(hfdata0, hfdata1, op):
    ''' Read data in from hdf5 in chunks, performing op, then returning an array of results.'''
    res = []

    for i in tqdm(range(0, hfdata0.shape[0], chunk_size)):
        a = np.array(hfdata0[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
        b = np.array(hfdata1[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
        c = op(a, b)

        for j in range(0, chunk_size):
            res.append(float(c[j]))

    return res


def do_stats(args):
    ''' Perform the statistics on the HDF5 data.'''
    from scipy.stats import spearmanr, pearsonr
    from tabulate import tabulate
    from rich.pretty import pprint
    from rich import print

    with h5py.File(args.load + ".h5", 'r') as hf:
        asi_actual_hf = hf['asi_actual']
        asj_actual_hf = hf['asj_actual']
        asi_pred_hf = hf['asi_pred']
        asj_pred_hf = hf['asj_pred']
        og_hf = hf['og']
        og_back_hf = hf['og_back']
        back = np.array(hf['back'])

        # Sum operation to perform
        sum_op = lambda x, y: np.sum(x * y, axis=(1,2,3))

        pprint("Counts Correlation - Reading Chunks")
        asi_real_count = np.array(read_chunks(asi_actual_hf, og_hf, sum_op))
        asj_real_count = np.array(read_chunks(asj_actual_hf, og_hf, sum_op))
        asi_pred_count = np.array(read_chunks(asi_pred_hf, og_hf, sum_op))
        asj_pred_count = np.array(read_chunks(asj_pred_hf, og_hf, sum_op))

        print("Set size:", len(asi_real_count))

        pprint("Correlations - spearmans & pearsons - ASI, ASJ - no background removal")
        asi_combo_spear = spearmanr(asi_real_count, asi_pred_count)
        asj_combo_spear = spearmanr(asj_real_count, asj_pred_count)
        asi_combo_pear = pearsonr(asi_real_count, asi_pred_count)
        asj_combo_pear = pearsonr(asj_real_count, asj_pred_count)

        table0 = [["Neuron", "Spearman", "P-Value", "Pearson", "P-Value"],
            ["ASI", asi_combo_spear[0], asi_combo_spear[1], asi_combo_pear[0], asi_combo_pear[1]],
            ["ASJ", asj_combo_spear[0], asj_combo_spear[1], asj_combo_pear[0], asj_combo_pear[1]]]
        	
        print(tabulate(table0, headers='firstrow'))

        # Now look at background removal
        pprint("Counts Correlation background removal - Reading Chunks")
        asi_real_count_back = np.array(read_chunks(asi_actual_hf, og_back_hf, sum_op))
        asi_pred_count_back = np.array(read_chunks(asi_pred_hf, og_back_hf, sum_op))
        asj_real_count_back = np.array(read_chunks(asj_actual_hf, og_back_hf, sum_op))
        asj_pred_count_back = np.array(read_chunks(asj_pred_hf, og_back_hf, sum_op))

        pprint("Correlations - spearmans & pearsons - ASI, ASJ - background removed")
        
        asi_combo_spear = spearmanr(asi_real_count_back, asi_pred_count_back)
        asj_combo_spear = spearmanr(asj_real_count_back, asj_pred_count_back)
        asi_combo_pear = pearsonr(asi_real_count_back, asi_pred_count_back)
        asj_combo_pear = pearsonr(asj_real_count_back, asj_pred_count_back)

        table1 = [["Neuron", "Spearman", "P-Value", "Pearson", "P-Value"],
            ["ASI", asi_combo_spear[0], asi_combo_spear[1], asi_combo_pear[0], asi_combo_pear[1]],
            ["ASJ", asj_combo_spear[0], asj_combo_spear[1], asj_combo_pear[0], asj_combo_pear[1]]]

        print(tabulate(table1, headers='firstrow'))


        # Anything this score or above is in the 95th percentile
        p95 = 291

        # Have a look at the border areas
        pprint("Analysing the Border Pixels")

        border_asi_values_actual = []
        border_asj_values_actual = []
        border_asi_values_pred = []
        border_asj_values_pred = []

        border_asi_95_actual = []
        border_asj_95_actual = []
        border_asi_95_pred = []
        border_asj_95_pred = []

        # Find the border
        for i in tqdm(range(0, asi_pred_hf.shape[0], chunk_size)):

            asi_pred_chunk = np.array(asi_pred_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asj_pred_chunk = np.array(asj_pred_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asi_actual_chunk = np.array(asi_actual_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asj_actual_chunk = np.array(asj_actual_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            og_chunk = np.array(og_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)

            for j in range(0, chunk_size):

                asi_pred_single = asi_pred_chunk[j].reshape(1, image_depth, image_height, image_width)
                asj_pred_single = asj_pred_chunk[j].reshape(1, image_depth, image_height, image_width)
                asi_actual_single = asi_actual_chunk[j].reshape(1, image_depth, image_height, image_width)
                asj_actual_single = asj_actual_chunk[j].reshape(1, image_depth, image_height, image_width)
                og = og_chunk[j].reshape(1, image_depth, image_height, image_width)

                border_image_asi_pred = find_border(asi_pred_single)
                border_image_asj_pred = find_border(asj_pred_single)

                border_image_asi_actual = find_border(asi_actual_single)
                border_image_asj_actual = find_border(asj_actual_single)
            
                tt = (border_image_asi_pred * og).flatten(); tt = tt[tt != 0]; border_asi_values_pred = border_asi_values_pred + list(tt)
                tt = (border_image_asj_pred * og).flatten(); tt = tt[tt != 0]; border_asj_values_pred = border_asj_values_pred + list(tt)
                tt = (border_image_asi_actual * og).flatten(); tt = tt[tt != 0]; border_asi_values_actual = border_asi_values_actual + list(tt)
                tt = (border_image_asj_actual * og).flatten(); tt = tt[tt != 0]; border_asj_values_actual = border_asj_values_actual + list(tt)

                border_asi_95_actual.append( (np.sum(np.where((border_image_asi_actual * og) > p95, 1, 0)) / np.sum(border_image_asi_actual)) * 100.0)
                border_asj_95_actual.append( (np.sum(np.where((border_image_asj_actual * og) > p95, 1, 0)) / np.sum(border_image_asj_actual)) * 100.0)
                border_asi_95_pred.append( (np.sum(np.where((border_image_asi_pred * og) > p95, 1, 0)) / np.sum(border_image_asi_pred)) * 100.0)
                border_asj_95_pred.append( (np.sum(np.where((border_image_asj_pred * og) > p95, 1, 0)) / np.sum(border_image_asj_pred)) * 100.0)

                # Can save out the images if we so desire, to check it's all working
                #save_fits(border_image_asi, "border_asi_" + str(idx) + ".fits")
                #save_fits(border_image_asj, "border_asj_" + str(idx) + ".fits")
                #save_fits(asi_pred_single, "asi_pred_" + str(idx) + ".fits")
                #save_fits(asi_actual_single, "asi_actual_" + str(idx) + ".fits")
                #save_fits(asj_pred_single, "asj_" + str(idx) + ".fits")
        
        table2 = [["Neuron", "Mean", "Median", "Standard", "Above 95th Median"],
            ["ASI Original", np.mean(border_asi_values_actual), np.median(border_asi_values_actual), np.std(border_asi_values_actual), np.median(border_asi_95_actual)],
            ["ASJ Original", np.mean(border_asj_values_actual), np.median(border_asj_values_actual), np.std(border_asj_values_actual), np.median(border_asj_95_actual)],
            ["ASI Predicted", np.mean(border_asi_values_pred), np.median(border_asi_values_pred), np.std(border_asi_values_pred), np.median(border_asi_95_pred)],
            ["ASJ Predicted", np.mean(border_asj_values_pred), np.median(border_asj_values_pred), np.std(border_asj_values_pred), np.median(border_asj_95_pred)]]

        print(tabulate(table2, headers='firstrow'))

        # Look at the false postives and false negatives, the areas of the predicted and original sizes and the scores above 95th percentile
        pprint("Analysing the False Positive and False Negative Voxels")

        false_op = lambda x, y: y * np.where(x == 1, 0, 1)
        count_asi_false_pos = []
        count_asj_false_pos = []
        count_asi_false_neg = []
        count_asj_false_neg = []

        area_asi_false_pos = []
        area_asj_false_pos = []
        area_asi_false_neg = []
        area_asj_false_neg = []

        area_asi_pred = []
        area_asj_pred = []
        area_asi_actual = []
        area_asj_actual = []

        asi_95_actual = []
        asj_95_actual = []
        asi_95_pred = []
        asj_95_pred = []

        asi_fp_95 = []
        asj_fp_95 = []
        asi_fn_95 = []
        asj_fn_95 = []

        for i in tqdm(range(0, asi_pred_hf.shape[0], chunk_size)):
            asi_pred = np.array(asi_pred_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asj_pred = np.array(asj_pred_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asi_actual = np.array(asi_actual_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asj_actual = np.array(asj_actual_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            og_chunk = np.array(og_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)

            asi_false_pos = false_op(asi_actual, asi_pred)
            asj_false_pos = false_op(asj_actual, asj_pred)
            asi_false_neg = false_op(asi_pred, asi_actual)
            asj_false_neg = false_op(asj_pred, asj_actual)

            for j in range(0, chunk_size):
                asi_95_actual.append( (np.sum(np.where((asi_actual[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asi_actual[j])) * 100.0)
                asj_95_actual.append( (np.sum(np.where((asj_actual[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asj_actual[j])) * 100.0)
                asi_95_pred.append( (np.sum(np.where((asi_pred[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asi_pred[j])) * 100.0)
                asj_95_pred.append( (np.sum(np.where((asj_pred[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asj_pred[j])) * 100.0)

                asi_fp_95.append( (np.sum(np.where((asi_false_pos[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asi_false_pos[j])) * 100.0)
                asj_fp_95.append( (np.sum(np.where((asj_false_pos[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asj_false_pos[j])) * 100.0)
                asi_fn_95.append( (np.sum(np.where((asi_false_neg[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asi_false_neg[j])) * 100.0)
                asj_fn_95.append( (np.sum(np.where((asj_false_neg[j] * og_chunk[j]) > p95, 1, 0)) / np.sum(asj_false_neg[j])) * 100.0)

            area = np.sum(asi_pred); area = list(area.flatten()); area_asi_pred = area_asi_pred + area
            area = np.sum(asj_pred); area = list(area.flatten()); area_asj_pred = area_asj_pred + area
            area = np.sum(asi_actual); area = list(area.flatten()); area_asi_actual = area_asi_actual + area
            area = np.sum(asj_actual); area = list(area.flatten()); area_asj_actual = area_asj_actual + area

            area = np.sum(asi_false_pos); area = list(area.flatten()); area_asi_false_pos = area_asi_false_pos + area
            area = np.sum(asj_false_pos); area = list(area.flatten()); area_asj_false_pos = area_asj_false_pos + area
            area = np.sum(asi_false_neg); area = list(area.flatten()); area_asi_false_neg = area_asi_false_neg + area
            area = np.sum(asj_false_neg); area = list(area.flatten()); area_asj_false_neg = area_asj_false_neg + area

            asi_false_pos = asi_false_pos * og_chunk; asi_false_pos = asi_false_pos.flatten()
            asj_false_pos = asj_false_pos * og_chunk; asj_false_pos = asj_false_pos.flatten()
            asi_false_neg = asi_false_neg * og_chunk; asi_false_neg = asi_false_neg.flatten()
            asj_false_neg = asj_false_neg * og_chunk; asj_false_neg = asj_false_neg.flatten()

            asi_false_pos = asi_false_pos[asi_false_pos !=0]
            asj_false_pos = asj_false_pos[asj_false_pos !=0]
            asi_false_neg = asi_false_neg[asi_false_neg !=0]
            asj_false_neg = asj_false_neg[asj_false_neg !=0]

            count_asi_false_pos = count_asi_false_pos + list(asi_false_pos)
            count_asj_false_pos = count_asj_false_pos + list(asj_false_pos)
            count_asi_false_neg = count_asi_false_neg + list(asi_false_neg)
            count_asj_false_neg = count_asj_false_neg + list(asj_false_neg)

        table3 = [["Neuron", "Mean", "Median", "Standard"],
            ["ASI False Pos", np.mean(count_asi_false_pos), np.median(count_asi_false_pos), np.std(count_asi_false_pos)],
            ["ASJ False Pos", np.mean(count_asj_false_pos), np.median(count_asj_false_pos), np.std(count_asj_false_pos)],
            ["ASI False Neg", np.mean(count_asi_false_neg), np.median(count_asi_false_neg), np.std(count_asi_false_neg)],
            ["ASJ False Neg", np.mean(count_asj_false_neg), np.median(count_asj_false_neg), np.std(count_asj_false_neg)]]

        print(tabulate(table3, headers='firstrow'))

        table4 = [["Neuron", "Mean", "Median", "Standard", "Above 95th Median"],
            ["ASI Pred Areas", np.mean(area_asi_pred), np.median(area_asi_pred), np.std(area_asi_pred), np.median(asi_95_pred)],
            ["ASJ Pred Areas", np.mean(area_asj_pred), np.median(area_asj_pred), np.std(area_asj_pred), np.median(asj_95_pred)],
            ["ASI Actual Areas", np.mean(area_asi_actual), np.median(area_asi_actual), np.std(area_asi_actual), np.median(asi_95_actual)],
            ["ASJ Acutal Areas", np.mean(area_asj_actual), np.median(area_asj_actual), np.std(area_asj_actual), np.median(asj_95_actual)],
            ["ASI False Pos Areas", np.mean(area_asi_false_pos), np.median(area_asi_false_pos), np.std(area_asi_false_pos), np.median(asi_fp_95)],
            ["ASJ False Pos Areas", np.mean(area_asj_false_pos), np.median(area_asj_false_pos), np.std(area_asj_false_pos), np.median(asj_fp_95)],
            ["ASI False Neg Areas", np.mean(area_asi_false_neg), np.median(area_asi_false_neg), np.std(area_asi_false_neg), np.median(asi_fn_95)],
            ["ASJ False Neg Areas", np.mean(area_asj_false_neg), np.median(area_asj_false_neg), np.std(area_asj_false_neg), np.median(asj_fn_95)]]

        print(tabulate(table4, headers='firstrow'))


        # Now have a quick look at the scores in each area
        pprint("Average scores in original and predicted areas")

        scores_asi_pred = []
        scores_asj_pred = []
        scores_asi_actual = []
        scores_asj_actual = []

        for i in tqdm(range(0, asi_pred_hf.shape[0], chunk_size)):

            asi_pred_chunk = np.array(asi_pred_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asj_pred_chunk = np.array(asj_pred_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asi_actual_chunk = np.array(asi_actual_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            asj_actual_chunk = np.array(asj_actual_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)
            og_chunk = np.array(og_hf[i: i + chunk_size]).reshape(chunk_size,image_depth, image_height, image_width)

            for j in range(0, chunk_size):

                asi_pred_single = asi_pred_chunk[j].reshape(1, image_depth, image_height, image_width)
                asj_pred_single = asj_pred_chunk[j].reshape(1, image_depth, image_height, image_width)
                asi_actual_single = asi_actual_chunk[j].reshape(1, image_depth, image_height, image_width)
                asj_actual_single = asj_actual_chunk[j].reshape(1, image_depth, image_height, image_width)
                og = og_chunk[j].reshape(1, image_depth, image_height, image_width)

                tt = og * asi_pred_single; tt = tt.flatten(); tt = tt[tt !=0]; scores_asi_pred = scores_asi_pred + list(tt)
                tt = og * asj_pred_single; tt = tt.flatten(); tt = tt[tt !=0]; scores_asj_pred = scores_asj_pred + list(tt)
                tt = og * asi_actual_single; tt = tt.flatten(); tt = tt[tt !=0]; scores_asi_actual = scores_asi_actual + list(tt)
                tt = og * asj_actual_single; tt = tt.flatten(); tt = tt[tt !=0]; scores_asj_actual = scores_asj_actual + list(tt)


        table5 = [["Neuron", "Mean", "Median", "Standard"],
            ["ASI Pred Values", np.mean(scores_asi_pred), np.median(scores_asi_pred), np.std(scores_asi_pred)],
            ["ASJ Pred Values", np.mean(scores_asj_pred), np.median(scores_asj_pred), np.std(scores_asj_pred)],
            ["ASI Actual Values", np.mean(scores_asi_actual), np.median(scores_asi_actual), np.std(scores_asi_actual)],
            ["ASJ Acutal Values", np.mean(scores_asj_actual), np.median(scores_asj_actual), np.std(scores_asj_actual)]]

        print(tabulate(table5, headers='firstrow'))


        '''
        # Lets take a look at the entropy in these areas
        asi_false_pos = np.array(asi_false_pos_hf)
        asi_false_neg = np.array(asi_false_neg_hf)
        asj_false_pos = np.array(asj_false_pos_hf)
        asj_false_neg = np.array(asj_false_neg_hf)

        # Base probability of the predicted regions
        asi_actual = np.array(asi_actual_hf)
        pa = asi_actual[asi_actual != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        asi_actual_prob = counts / len(pa)
        asi_actual_prob = np.where(asi_actual_prob <= 0, 1e-10, asi_actual_prob) # Helps with KL - there must always be a chance of success

        asj_actual =  np.array(asj_actual_hf)
        pa = asj_actual[asj_actual != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        asj_actual_prob = counts / len(pa)
        asj_actual_prob = np.where(asj_actual_prob <= 0, 1e-10, asj_actual_prob) # Helps with KL - there must always be a chance of success

        # Base probability of the predicted regions
        asi_pred =  np.array(asi_pred_hf)
        pa = asi_pred[asi_pred != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        asi_pred_prob = counts / len(pa)
        asi_pred_prob = np.where( asi_pred_prob <= 0, 1e-10, asi_pred_prob) # Helps with KL - there must always be a chance of success

        asj_pred =  np.array(asj_pred_hf)
        pa = asj_pred[asj_pred != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        asj_pred_prob = counts / len(pa)
        asj_pred_prob = np.where( asj_pred_prob <= 0, 1e-10, asj_pred_prob) # Helps with KL - there must always be a chance of success

        kldiv = sum(scipy.special.kl_div(asi_actual_prob, asi_pred_prob))
        jensen = scipy.spatial.distance.jensenshannon(asi_actual_prob, asi_pred_prob)
        print("ASI actual to pred KL-Div, Jensen", kldiv, jensen)

        kldiv = sum(scipy.special.kl_div(asj_actual_prob, asj_pred_prob))
        jensen = scipy.spatial.distance.jensenshannon(asj_actual_prob, asj_pred_prob)
        print("ASJ actual to pred KL-Div", kldiv, jensen)
        
        pa = asi_false_pos[asi_false_pos != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        probs = counts / len(pa)
        probs = np.where( probs <= 0, 1e-10, probs) 
        #asi_false_pos_shannon = scipy.stats.entropy(counts, base=len(counts))
        #print("ASI false Pos Shannon: ", asi_false_pos_shannon)
        kldiv = sum(scipy.special.kl_div(asi_actual_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asi_actual_prob, probs)
        print("ASI Actual to false Pos KL-Div, Jensen", kldiv, jensen)
        kldiv = sum(scipy.special.kl_div(asi_pred_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asi_pred_prob, probs)
        print("ASI Pred to false pos KL-Div, Jensen", kldiv, jensen)
        
        pa = asi_false_neg[asi_false_neg != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        probs = counts / len(pa)
        probs = np.where( probs <= 0, 1e-10, probs)
        kldiv = sum(scipy.special.kl_div(asi_actual_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asi_actual_prob, probs)
        print("ASI Actual to false Neg KL-Div, Jensen", kldiv, jensen)
        kldiv = sum(scipy.special.kl_div(asi_pred_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asi_pred_prob, probs)
        print("ASI Pred to false Neg KL-Div, Jensen", kldiv, jensen)

        pa = asj_false_pos[asj_false_pos != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        probs = counts / len(pa)
        probs = np.where( probs <= 0, 1e-10, probs)
        kldiv = sum(scipy.special.kl_div(asj_actual_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asj_actual_prob, probs)
        print("ASJ Pred to false Pos KL-Div, Jensen", kldiv, jensen)
        kldiv = sum(scipy.special.kl_div(asj_pred_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asj_pred_prob, probs)
        print("ASJ Actual to false Pos KL-Div, Jensen", kldiv, jensen)

        pa = asj_false_neg[asj_false_neg != 0].astype(np.int32)
        counts = np.bincount(pa, minlength=4096)
        probs = counts / len(pa)
        probs = np.where( probs <= 0, 1e-10, probs)
        kldiv = sum(scipy.special.kl_div(asj_pred_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asj_pred_prob, probs)
        print("ASJ Pred to false Neg KL-Div, Jensen", kldiv, jensen)
        kldiv = sum(scipy.special.kl_div(asj_actual_prob, probs))
        jensen = scipy.spatial.distance.jensenshannon(asj_actual_prob, probs)
        print("ASJ Actual to false Neg KL-Div, Jensen", kldiv, jensen)

        # Lets look at how divided ASI is from ASJ
        kldiv = sum(scipy.special.kl_div(asi_actual_prob, asj_actual_prob))
        jensen = scipy.spatial.distance.jensenshannon(asi_actual_prob, asj_actual_prob)
        print("ASI to ASJ Actual  KL-Div, Jensen", kldiv, jensen)

        kldiv = sum(scipy.special.kl_div(asi_pred_prob, asj_pred_prob))
        jensen = scipy.spatial.distance.jensenshannon(asi_pred_prob, asj_pred_prob)
        print("ASI to ASJ Pred  KL-Div, Jensen", kldiv, jensen)
        '''

        '''
        # Plot the Predicted against the Original
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches((8,6))

        individuals = list(range(len(asi_real_count)))
        df0 = pd.DataFrame({"individual": individuals, "asi_real": asi_real_count, "asi_pred": asi_pred_count})
        df1 = pd.DataFrame({"individual": individuals, "asj_real": asj_real_count, "asj_pred": asj_pred_count})
    
        axes[0].xaxis.set_label_text("asi base luminance")
        axes[1].xaxis.set_label_text("asj base luminance")
    
        axes[0].yaxis.set_label_text("asi pred luminance")
        axes[1].yaxis.set_label_text("asj pred luminance")


        sns.scatterplot(data=df0, x="asi_real", y="asi_pred", ax=axes[0])
        sns.scatterplot(data=df1, x="asj_real", y="asj_pred", ax=axes[1])
        
        plt.savefig(args.save + 'asi_vs_asj.png')
        
        # Plot the Predicted against the Original with background removal
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches((8,6))

        individuals = list(range(len(asi_real_count)))
        df0 = pd.DataFrame({"individual": individuals, "asi_real": asi_real_count_back, "asi_pred": asi_pred_count_back})
        df1 = pd.DataFrame({"individual": individuals, "asj_real": asj_real_count_back, "asj_pred": asj_pred_count_back})
    
        axes[0].xaxis.set_label_text("asi base luminance")
        axes[1].xaxis.set_label_text("asj base luminance")
    
        axes[0].yaxis.set_label_text("asi pred luminance")
        axes[1].yaxis.set_label_text("asj pred luminance")

        sns.scatterplot(data=df0, x="asi_real", y="asi_pred", ax=axes[0])
        sns.scatterplot(data=df1, x="asj_real", y="asj_pred", ax=axes[1])
        
        plt.savefig(args.save + 'asi_vs_asj_back.png')
        '''


def csv_stats(csv_path):
    """ If we have already computed the stats, lets present some averages 
    and visualise. """
    import csv
    from tabulate import tabulate
    from rich.pretty import pprint
    from rich import print
    
    scores  = []

    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for line in reader:
            scores.append(line)

    asi_jacc = []
    asj_jacc = []
    all_jacc = []

    asi_fp = []
    asj_fp = []
    asi_fn = []
    asj_fn = []

    for score in scores:
        asi_jacc.append(float(score['Jacc1']))
        asj_jacc.append(float(score['Jacc2']))
        all_jacc.append(float(score['Jaccard']))
        
        asi_fp.append(float(score['fp1']))
        asj_fp.append(float(score['fp2']))
        asi_fn.append(float(score['fn1']))
        asj_fn.append(float(score['fn2']))

    asi_jacc = np.array(asi_jacc)
    asj_jacc = np.array(asj_jacc)
    all_jacc = np.array(all_jacc)

    asi_fp = np.array(asi_fp)
    asj_fp = np.array(asj_fp)
    asi_fn = np.array(asi_fn)
    asj_fn = np.array(asj_fn)

    pprint("Jaccard Scores")
    table0 = [ ["Neuron", "Min", "Max", "Mean", "Median", "Std.Dev"],
            ["ASI", min(asi_jacc),  max(asi_jacc),  np.mean(asi_jacc), np.median(asi_jacc), np.std(asi_jacc)],
            ["ASJ", min(asj_jacc),  max(asj_jacc),  np.mean(asj_jacc), np.median(asj_jacc), np.std(asj_jacc)],
            ["Both", min(all_jacc),  max(all_jacc),  np.mean(all_jacc), np.median(all_jacc), np.std(all_jacc)]]
        	
    print(tabulate(table0, headers='firstrow'))

    pprint("Number of Jaccard scores below 0.5 for each neuron")
    table1 = [["Neuron", "Num scores below 0.5"],
        ["ASI", len(asi_jacc[asi_jacc < 0.5])],
        ["ASJ", len(asi_jacc[asj_jacc < 0.5])]]
    
    print(tabulate(table1, headers='firstrow'))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="U-Net Data Analysis")

    parser.add_argument('--dataset', default="")
    parser.add_argument('--savedir', default=".")
    parser.add_argument('--base', default="")
    parser.add_argument('--rep', default="")
    parser.add_argument('--save', default="summary_stats", help="Name WITHOUT extension for the saved output files.")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument('--load', default="summary_stats", help="Filename WITHOUT extension for the .h5 and csv to process.")

    args = parser.parse_args()
    data = None

    if args.dataset != "":
        sources_masks, og_sources, og_masks, rois  = find_image_pairs(args)
        read_counts(args, sources_masks, og_sources, og_masks, rois)

    elif os.path.exists(args.load + ".h5"):
        do_stats(args)
        csv_stats(args.load + ".csv")
        
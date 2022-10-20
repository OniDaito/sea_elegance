
""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

stats.py - look at the worm data and generate some stats

Example use:
python unet_stats.py --base /phd/wormz/queelim --rep /media/proto_backup/wormz/queelim --dataset /media/proto_backup/wormz/queelim/dataset_3d_basic_noresize --savedir /media/proto_working/runs/wormz_2022_09_19 --no-cuda --nclasses 3
python unet_stats.py --load data.pickle --nclasses 3

"""

import pickle
import torch
import numpy as np
import argparse
import csv
import os
import torch.nn as nn
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from util.loadsave import load_model, load_checkpoint
from util.image import load_fits, save_fits
import torch.nn.functional as F


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
        print(idx)
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
    
    return sources_masks, og_sources, og_masks, rois

def crop_image(image, roi, target_size):
    ''' Crop the image. However the crop might be bigger than the actual
    target size due to '''
    xs = roi['xs']
    xe = roi['xe']
    ys = roi['ys']
    ye = roi['ye']
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

    asi_1_pred = []
    asi_2_pred = []
    asj_1_pred = []
    asj_2_pred = []

    asi_1_total = []
    asi_2_total = []
    asj_1_total = []
    asj_2_total = []

    asi_1_total_pred = []
    asi_2_total_pred = []

    asj_1_total_pred = []
    asj_2_total_pred = []

    asi_1_total_false_pos = []
    asi_2_total_false_pos = []

    asi_1_total_false_neg = []
    asi_2_total_false_neg = []

    asj_1_total_false_pos = []
    asj_2_total_false_pos = []

    asj_1_total_false_neg = []
    asj_2_total_false_neg = []

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
                assert(not (torch.all(prediction == 0).item()))

                classes = prediction[0].detach().cpu().squeeze()
                classes = F.one_hot(classes.argmax(dim=0), args.nclasses).permute(3, 0, 1, 2)
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

                #input_image = torch.narrow(input_image, 0, 0, resized_prediction.shape[0])
                print("Shapes", classes.shape, input_image.shape)

                if args.nclasses == 5:

                    asi_1_pred = torch.where(resized_prediction == 1, 1, 0)
                    asi_2_pred = torch.where(resized_prediction == 2, 1, 0)

                    asj_1_pred = torch.where(resized_prediction == 3, 1, 0)
                    asj_2_pred = torch.where(resized_prediction == 4, 1, 0)

                    asi_1_mask = torch.where(target_image == 1, 1, 0)
                    asi_2_mask = torch.where(target_image == 2, 1, 0)
                    asj_1_mask = torch.where(target_image == 3, 1, 0)
                    asj_2_mask = torch.where(target_image == 4, 1, 0)

                    count_asi_1 = float(torch.sum(asi_1_pred * og_image))
                    count_asi_2 = float(torch.sum(asi_2_pred * og_image))
                    count_asj_1 = float(torch.sum(asj_1_pred * og_image))
                    count_asj_2 = float(torch.sum(asj_2_pred * og_image))

                    count_asi_1_real = float(torch.sum(asi_1_mask * og_image))
                    count_asi_2_real = float(torch.sum(asi_2_mask * og_image))
                    count_asj_1_real = float(torch.sum(asj_1_mask * og_image))
                    count_asj_2_real = float(torch.sum(asj_2_mask * og_image))

                    asi_1_total.append(count_asi_1_real)
                    asi_2_total.append(count_asi_2_real)
                    asj_1_total.append(count_asj_1_real)
                    asj_2_total.append(count_asj_2_real)

                    print("Counts from OG image", count_asi_1, count_asi_2, count_asj_1, count_asj_2)

                    asi_1_total_pred.append(count_asi_1)
                    asi_2_total_pred.append(count_asi_2)

                    asj_1_total_pred.append(count_asj_1)
                    asj_2_total_pred.append(count_asj_2)

                    # Now look at the false pos, false neg and get the scores
                    asi_1_pred_inv = torch.where(resized_prediction == 1, 0, 1)
                    asi_2_pred_inv = torch.where(resized_prediction == 2, 0, 1)
                    asj_1_pred_inv = torch.where(resized_prediction == 3, 0, 1)
                    asj_2_pred_inv = torch.where(resized_prediction == 4, 0, 1)

                    asi_1_mask_inv = torch.where(target_image == 1, 0, 1)
                    asi_2_mask_inv = torch.where(target_image == 2, 0, 1)
                    asj_1_mask_inv = torch.where(target_image == 3, 0, 1)
                    asj_2_mask_inv = torch.where(target_image == 4, 0, 1)

                    asi_1_false_pos = asi_1_mask_inv * asi_1_pred
                    asi_2_false_pos = asi_2_mask_inv * asi_2_pred
                    asj_1_false_pos = asj_1_mask_inv * asj_1_pred
                    asj_2_false_pos = asj_2_mask_inv * asj_2_pred

                    asi_1_false_neg = asi_1_mask * asi_1_pred_inv
                    asi_2_false_neg = asi_2_mask * asi_2_pred_inv
                    asj_1_false_neg = asj_1_mask * asj_1_pred_inv
                    asj_2_false_neg = asj_2_mask * asj_2_pred_inv

                    count_asi_1_false_pos = asi_1_false_pos * og_image
                    count_asi_2_false_pos = asi_2_false_pos * og_image
                    count_asi_1_false_neg = asi_1_false_neg * og_image
                    count_asi_2_false_neg = asi_2_false_neg * og_image

                    count_asj_1_false_pos = asj_1_false_pos * og_image
                    count_asj_2_false_pos = asj_2_false_pos * og_image
                    count_asj_1_false_neg = asj_1_false_neg * og_image
                    count_asj_2_false_neg = asj_2_false_neg * og_image
        
                    asi_1_total_false_pos.append(count_asi_1_false_pos)
                    asi_2_total_false_pos.append(count_asi_2_false_pos)

                    asi_1_total_false_neg.append(count_asi_1_false_neg)
                    asi_2_total_false_neg.append(count_asi_2_false_neg)

                    asj_1_total_false_pos.append(count_asj_1_false_pos)
                    asj_2_total_false_pos.append(count_asj_2_false_pos)

                    asj_1_total_false_neg.append(count_asj_1_false_neg)
                    asj_2_total_false_neg.append(count_asj_2_false_neg)

                elif args.nclasses == 3:
                    asi_1_pred = torch.where(resized_prediction == 1, 1, 0)

                    asj_1_pred = torch.where(resized_prediction == 2, 1, 0)

                    asi_1_mask = torch.where(target_image == 1, 1, 0)
                    asj_1_mask = torch.where(target_image == 2, 1, 0)

                    count_asi_1 = asi_1_pred * og_image
                    count_asj_1 = asj_1_pred * og_image

                    count_asi_1_real = asi_1_mask * og_image
                    count_asj_1_real = asj_1_mask * og_image

                    asi_1_total.append(count_asi_1_real.numpy())
                    asj_1_total.append(count_asj_1_real.numpy())

                    print("Counts from OG image", count_asi_1, count_asj_1)

                    asi_1_total_pred.append(count_asi_1.numpy())
                    asj_1_total_pred.append(count_asj_1.numpy())

                    # Now look at the false pos, false neg and get the scores
                    asi_1_pred_inv = torch.where(resized_prediction == 1, 0, 1)
                    asj_1_pred_inv = torch.where(resized_prediction == 2, 0, 1)

                    asi_1_mask_inv = torch.where(target_image == 1, 0, 1)
                    asj_1_mask_inv = torch.where(target_image == 3, 0, 1)

                    asi_1_false_pos = asi_1_mask_inv * asi_1_pred
                    asj_1_false_pos = asj_1_mask_inv * asj_1_pred

                    asi_1_false_neg = asi_1_mask * asi_1_pred_inv
                    asj_1_false_neg = asj_1_mask * asj_1_pred_inv

                    count_asi_1_false_pos = asi_1_false_pos * og_image
                    count_asi_1_false_neg = asi_1_false_neg * og_image

                    count_asj_1_false_pos = asj_1_false_pos * og_image
                    count_asj_1_false_neg = asj_1_false_neg * og_image
        
                    asi_1_total_false_pos.append(count_asi_1_false_pos.numpy())
                    asi_1_total_false_neg.append(count_asi_1_false_neg.numpy())
                    asj_1_total_false_pos.append(count_asj_1_false_pos.numpy())
                    asj_1_total_false_neg.append(count_asj_1_false_neg.numpy())

    data = {}
    data["asi_1_actual"] = asi_1_total
    data["asi_2_actual"] = asi_2_total
    data["asj_1_actual"] = asj_1_total
    data["asj_2_actual"] = asj_2_total
    data["asi_1_pred"] = asi_1_total_pred
    data["asi_2_pred"] = asi_2_total_pred
    data["asj_1_pred"] = asj_1_total_pred
    data["asj_2_pred"] = asj_2_total_pred
    data["asi_1_false_pos"] = asi_1_total_false_pos
    data["asi_2_false_pos"] = asi_1_total_false_pos
    data["asi_1_false_neg"] = asi_1_total_false_neg
    data["asi_2_false_neg"] = asi_2_total_false_neg
    data["asj_1_false_pos"] = asj_1_total_false_pos
    data["asj_2_false_pos"] = asj_2_total_false_pos
    data["asj_1_false_neg"] = asj_1_total_false_neg
    data["asj_2_false_neg"] = asj_1_total_false_neg

    data["og_masks"] = og_masks
    data["sources"] = og_sources
    data["pairs"] = sources_masks

    with open('summary_stats.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data

def do_stats(args, data):
    ''' Now we have the data, lets do the stats on it.''' 

    if args.nclasses == 5:

        # Convert any tensors we have into values
        for i in range(len(data["asi_1_pred"])):
            data["asi_1_pred"][i] = float(data["asi_1_pred"][i])
            data["asi_2_pred"][i] = float(data["asi_2_pred"][i])
            data["asj_1_pred"][i] = float(data["asj_1_pred"][i])
            data["asj_2_pred"][i] = float(data["asj_2_pred"][i])

        asi_combo_real =  [data['asi_1_actual'][i] + data['asi_2_actual'][i] for i in range(len(data['asi_1_actual']))]
        asi_combo_pred =  [data['asi_1_pred'][i] + data['asi_2_pred'][i] for i in range(len(data['asi_1_actual']))]

        print("Correlations - spearmans & pearsons - ASI, ASJ")
        asi_combo_cor = spearmanr(asi_combo_real, asi_combo_pred)
        asj_combo_cor = spearmanr(asj_combo_real, asj_combo_pred)
        print(asi_combo_cor, asj_combo_cor)

        asi_combo_cor = pearsonr(asi_combo_real, asi_combo_pred)
        asj_combo_cor = pearsonr(asj_combo_real, asj_combo_pred)
        print(asi_combo_cor, asj_combo_cor)

        combo_real =  [data['asi_1_actual'][i] + data['asi_2_actual'][i] + data['asj_1_actual'][i] + data['asj_2_actual'][i] for i in range(len(data['asi_1_actual']))]
        combo_pred =  [data['asi_1_pred'][i] + data['asi_2_pred'][i] + data['asj_1_pred'][i] + data['asj_2_pred'][i] for i in range(len(data['asi_1_pred']))]

        print("Correlations - spearmans & pearsons - ALL")
        combo_cor0 = spearmanr(combo_real, combo_pred)
        combo_cor1 = pearsonr(combo_real, combo_pred)
        print(combo_cor0, combo_cor1)

        '''
        asi_1_false_pos = np.array(data["asi_1_false_pos"])
        asi_2_false_pos = np.array(data["asi_1_false_pos"])

        asi_1_false_neg = np.array(data["asi_1_false_neg"])
        asi_2_false_neg = np.array(data["asi_2_false_neg"])

        asj_1_false_pos = np.array(data["asj_1_false_pos"])
        asj_2_false_pos = np.array(data["asj_2_false_pos"])

        asj_1_false_neg = np.array(data["asj_1_false_neg"])
        asj_2_false_neg = np.array(data["asj_2_false_neg"])
        

        print("False Positives and Negative Luminances")
        print("ASI 1 FP min, max, mean, std", np.min(asi_1_false_pos), np.max(asi_1_false_pos), np.mean(asi_1_false_pos), np.std(asi_1_false_pos))
        print("ASI 2 FP min, max, mean, std", np.min(asi_2_false_pos), np.max(asi_2_false_pos), np.mean(asi_2_false_pos), np.std(asi_2_false_pos))

        print("ASI 1 FN min, max, mean, std", np.min(asi_1_false_neg), np.max(asi_1_false_neg), np.mean(asi_1_false_neg), np.std(asi_1_false_neg))
        print("ASI 2 FN min, max, mean, std", np.min(asi_2_false_neg), np.max(asi_2_false_neg), np.mean(asi_2_false_neg), np.std(asi_2_false_neg))

        print("ASJ 1 FP min, max, mean, std", np.min(asj_1_false_pos), np.max(asj_1_false_pos), np.mean(asj_1_false_pos), np.std(asj_1_false_pos))
        print("ASJ 2 FP min, max, mean, std", np.min(asj_2_false_pos), np.max(asj_2_false_pos), np.mean(asj_2_false_pos), np.std(asj_2_false_pos))

        print("ASJ 1 FN min, max, mean, std", np.min(asj_1_false_neg), np.max(asj_1_false_neg), np.mean(asj_1_false_neg), np.std(asj_1_false_neg))
        print("ASJ 2 FN min, max, mean, std", np.min(asj_2_false_neg), np.max(asj_2_false_neg), np.mean(asj_2_false_neg), np.std(asj_2_false_neg)) '''
    
    elif args.nclasses == 3:
        asi_combo_real =  np.array([float(np.sum(i)) for i in data['asi_1_actual']])
        asi_combo_pred =  np.array([float(np.sum(i)) for i in data['asi_1_pred']])
        asj_combo_real =  np.array([float(np.sum(i)) for i in data['asj_1_actual']])
        asj_combo_pred =  np.array([float(np.sum(i)) for i in data['asj_1_pred']])

        print("Correlations - spearmans & pearsons - ASI, ASJ")
        asi_combo_cor = spearmanr(asi_combo_real, asi_combo_pred)
        asj_combo_cor = spearmanr(asj_combo_real, asj_combo_pred)
        print(asi_combo_cor, asj_combo_cor)
        asi_combo_cor = pearsonr(asi_combo_real, asi_combo_pred)
        asj_combo_cor = pearsonr(asj_combo_real, asj_combo_pred)
        print(asi_combo_cor, asj_combo_cor)


        # We need to see the values in our predicted masks versus the original masks to see which dist is better
        sig_value = 291
        asi_actual = np.array(data["asi_1_actual"][0]).flatten()

        for fp in data["asi_1_actual"][1:]:
            tp = np.array(fp).flatten()
            asi_actual = np.concatenate((asi_actual, tp))

        asi_actual = np.delete(asi_actual, np.where(asi_actual == 0.0))

        asi_pred = np.array(data["asi_1_pred"][0]).flatten()

        for fp in data["asi_1_pred"][1:]:
            tp = np.array(fp).flatten()
            asi_pred = np.concatenate((asi_pred, tp))

        asi_pred = np.delete(asi_pred, np.where(asi_pred == 0.0))

        # Plot a histogram for ASJ
        fig, ax = plt.subplots()
        ax.hist(asi_actual, bins=100, label='ASI Base', alpha=.5, color='blue')
        ax.hist(asi_pred, bins=100,  label='ASI Predicted', alpha=.5, color='red')
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Count')
        ax.set_title("Histogram of intensity values in the original and predicted masks for ASI.")
        ax.legend()
        plt.show()

        above_actual = np.sum(np.where(asi_actual >= sig_value, 1, 0))
        below_actual = np.sum(np.where(asi_actual < sig_value, 1, 0))
        above_pred = np.sum(np.where(asi_pred >= sig_value, 1, 0))
        below_pred = np.sum(np.where(asi_pred < sig_value, 1, 0))
        print("ASI Counts above and below sig for base and pred", above_actual, below_actual, above_pred, below_pred )

        # Now go with ASJ
        asj_actual = np.array(data["asj_1_actual"][0]).flatten()

        for fp in data["asj_1_actual"][1:]:
            tp = np.array(fp).flatten()
            asj_actual = np.concatenate((asj_actual, tp))

        asj_actual = np.delete(asj_actual, np.where(asj_actual == 0.0))
        asj_pred = np.array(data["asj_1_pred"][0]).flatten()

        for fp in data["asj_1_pred"][1:]:
            tp = np.array(fp).flatten()
            asj_pred = np.concatenate((asj_pred, tp))

        asj_pred = np.delete(asj_pred, np.where(asj_pred == 0.0))

        # Plot a histogram for ASI
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots()
        ax.hist(asj_actual, bins=100, label='ASJ Base', alpha=.5, color='blue')
        ax.hist(asj_pred, bins=100,  label='ASJ Predicted', alpha=.5, color='red')
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Count')
        ax.set_title("Histogram of intensity values in the original and predicted masks for ASJ.")
        ax.legend()
        plt.show()

        above_actual = np.sum(np.where(asj_actual >= sig_value, 1, 0))
        below_actual = np.sum(np.where(asj_actual < sig_value, 1, 0))
        above_pred = np.sum(np.where(asj_pred >= sig_value, 1, 0))
        below_pred = np.sum(np.where(asj_pred < sig_value, 1, 0))
        print("ASJ Counts above and below sig for base and pred", above_actual, below_actual, above_pred, below_pred )

    # Multiclass alignments
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2)

    # Joined alignments
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2)

    individuals = list(range(len(data["asi_1_actual"])))
    df0 = pd.DataFrame({"individual": individuals, "asi_combo_real": asi_combo_real, "asi_combo_pred": asi_combo_pred})
    df1 = pd.DataFrame({"individual": individuals, "asj_combo_real": asj_combo_real, "asj_combo_pred": asj_combo_pred})
   
    axes[0].xaxis.set_label_text("individuals")
    axes[1].xaxis.set_label_text("individuals")
   
    axes[0].yaxis.set_label_text("luminance")
    axes[1].yaxis.set_label_text("luminance")

    sns.lineplot(x="individual", y='value', hue='variable', 
             data=pd.melt(df0, ['individual']), ax=axes[0])
  
    sns.lineplot(x="individual", y='value', hue='variable', 
             data=pd.melt(df1, ['individual']), ax=axes[1])
    
    plt.show()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="U-Net Data Analysis")

    parser.add_argument('--dataset', default="")
    parser.add_argument('--savedir', default=".")
    parser.add_argument('--base', default="")
    parser.add_argument('--rep', default="")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    
    parser.add_argument("--nclasses", type=int, default=5,
                        help="Number of classes this network predicts",
    )
    parser.add_argument('--load', default="")
    args = parser.parse_args()
    data = None

    sources_masks, og_sources, og_masks, rois  = find_image_pairs(args)

    if args.load != "" and os.path.exists(args.load):
        with open(args.load, 'rb') as f:
            data = pickle.load(f)
    else:
        data = read_counts(args, sources_masks, og_sources, og_masks, rois)

    from scipy.stats import spearmanr, pearsonr

    print ("Data size", len(data["asi_1_pred"]))

    do_stats(args, data)
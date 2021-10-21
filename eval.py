""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

eval.py - evaluate our network based on the validation set

"""

import torch
import argparse
import sys
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from util.loadsave import load_checkpoint, load_model
from util.image import load_fits, save_fits, save_image, resize_3d, reduce_result, finalise_result


def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


def load_image(image_path: str, normalise=False, dtype=torch.float32) -> torch.Tensor:
    input_image = load_fits(image_path, dtype=dtype)
    if normalise:
        normalised_image = input_image / 4095.0
        input_image = torch.tensor(normalised_image)

    return input_image


def visualise_scores(scores: list):
    # Mean of all the overlays
    num_scores = len(scores)
    m = np.array([s['overlay'] for s in scores])
    mean_score = np.mean(m)
    print("Mean Overlay Score", mean_score, "over", num_scores, "images, with std dev", np.std(m))

    classes = np.array([s['classes'] for s in scores])
    for cdx in range(3):
        print("Class", cdx)
        scores = np.array([c[cdx] for c in classes])
        means = []
        for tdx in range(3):
            means.append(np.mean([s[tdx] for s in scores]))
        print(means)


def compare_masks(original: np.ndarray, predicted:  np.ndarray):
    # L1 Sum absolute differences
    l1 = np.subtract(original, predicted)
    l1 = np.absolute(l1)
    l1 = np.sum(l1)

    # Mask overlay score
    tsize = np.shape(original)
    tsize = tsize[0] * tsize[1] * tsize[2]
    overlay = np.sum(np.where(original == predicted, 1, 0)) / tsize * 100

    # Class accuracies 0,1,2
    class_accuracy = []

    for cid in range(3):
        m = np.where(original == cid, 1, 0)
        scores = []

        for tid in range(3):
            n = np.where(predicted == tid, 1, 0)
            match = np.sum(m * n) / np.sum(m) * 100
            scores.append(match)

        class_accuracy.append(scores)

    return {"l1": l1, "overlay": overlay, "classes": class_accuracy}


def mask_check(model, device, save_path, valid_set):
    """
    Test our model 

    Parameters
    ----------
    model : NetU
        The saved, trained model

    device : torch.Device
        The device to run on (cpu or cuda)

    valid_set : dictionary
        Two lists of input images and target pairs. We will check
        them over and produce some stats

    Returns
    -------
    None
    """

    # Need to call model.eval() to set certain layers to eval mode. Ones
    # like dropout and what not
    scores = []

    for idx in range(1, int((valid_set.size) / 2)):
        source_path = save_path + "/" + valid_set.iloc[idx, 0]
        mask_path = save_path + "/" + valid_set.iloc[idx, 1]

        source_image = load_image(source_path, normalise=True)
        mask_image = load_image(mask_path, normalise=False, dtype=torch.int8)

        with torch.no_grad():
            model.eval()
            im = source_image.unsqueeze(dim=0).unsqueeze(dim=0)
            im = im.to(device)
            prediction = model.forward(im)
            prediction = finalise_result(prediction)

            # Now compare both the prediction and the mask
            score = compare_masks(mask_image.numpy(), prediction)
            scores.append(score)

    visualise_scores(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sea-elegance run")
    parser.add_argument("--load", default=".", help="Path to our model dir.")
    parser.add_argument("--data", default=".", help="Path to our dataset dir.")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.load and args.data and os.path.isfile(args.load + "/checkpoint.pth.tar") and os.path.isfile(args.load + "/dataset_valid.csv"):
        # Load the model and all the parameters for it
        model = load_model(args.load + "/model.tar")
        (model, _, _, _, _, prev_args, _) = load_checkpoint(
            model, args.load, "checkpoint.pth.tar", device
        )
        model = model.to(device)
        model.eval()

        # Now load the CSV for the validation set

        valid_paths = pd.read_csv(args.load + "/dataset_valid.csv", names=["source", "target"],
                                  converters={'source': strip,
                                              'target': strip})

        mask_check(model, device, args.data, valid_paths)

    else:
        print("--load must point to a run directory. --data must point to the dataset")
        sys.exit(0)

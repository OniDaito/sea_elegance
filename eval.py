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
import matplotlib
import matplotlib.cm
from astropy.io import fits
import torch.nn.functional as F
from util.loadsave import load_checkpoint, load_model
from util.image import load_fits, reduce_source, save_image, reduce_result, finalise_result


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


def visualise_scores(score_path: list):
    scores = [s[0] for s in score_path]
    paths = [s[1] for s in score_path]
    # Mean of all the overlays
    num_scores = len(scores)
    m = np.array([s['overlay'] for s in scores])
    mean_score = np.mean(m)
    print("Mean Overlay Score", mean_score, "over",
          num_scores, "images, with std dev", np.std(m))
    m = np.array([s['overlap'] for s in scores])
    mean_score = np.mean(m)
    print("Mean Overlap Percentage: ", mean_score,  "over",
          num_scores, "images, with std dev", np.std(m))
    m = np.array([s['dice'] for s in scores])
    mean_score = np.mean(m)
    print("Mean Dice : ", mean_score,  "over",
          num_scores, "images, with std dev", np.std(m))
    m = np.array([s['jacc'] for s in scores])
    mean_score = np.mean(m)
    print("Mean Jaccard Score : ", mean_score,  "over",
          num_scores, "images, with std dev", np.std(m))

    for c in range(2):
        d = np.array([s['class_dice'][c] for s in scores])
        mean_score = np.mean(d)
        print("Mean Dice Score on Class", c + 1, " : ", mean_score, "over",
            num_scores, "images, with std dev", np.std(d))

    for i, s in enumerate(scores):
        print(i, " - ", paths[i], " - Jacc:", s['jacc'], " Dice:", s['dice'])


def compare_masks(original: np.ndarray, predicted:  np.ndarray):
    # L1 Sum absolute differences
    l1 = np.subtract(original, predicted)
    l1 = np.absolute(l1)
    l1 = np.sum(l1)

    # Mask overlay score
    tsize = np.shape(original)
    tsize = tsize[0] * tsize[1] * tsize[2]
    overlay = np.sum(np.where(original == predicted, 1, 0)) / tsize * 100

    # Of the areas, how did we get it right?
    m = np.where(original != 0, 1, 0)
    n = np.where(predicted != 0, 1, 0)
    overlap = np.sum(m * n) / np.sum(original) * 100

    # Dice score
    dice = (2 * np.sum(m * n)) / (np.sum(m) + np.sum(n))

    # Jaccard
    jacc = np.sum(n * m) / (np.sum(n) + np.sum(m) - np.sum(n * m))

    # Per class Dice and Jaccard
    class_dice = []
    class_jacc = []

    for c in range(1, 3):
        m = np.where(original == c, 1, 0)
        n = np.where(predicted == c, 1, 0)
        dice = (2 * np.sum(m * n)) / (np.sum(m) + np.sum(n))
        jacc = np.sum(n * m) / (np.sum(n) + np.sum(m) - np.sum(n * m))
        class_dice.append(dice)
        class_jacc.append(jacc)

    return {"l1": l1, "overlay": overlay, "overlap": overlap,
            "dice": dice, "jacc": jacc, "class_dice": class_dice, "class_jacc": class_jacc}


def colourize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colourize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = np.amin(value) if vmin is None else vmin
    vmax = np.amax(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = np.squeeze(value)

    # quantize
    indices = np.round(value * 255).astype(np.int32)

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'viridis')
    colours = np.array(cm.colors, dtype=np.float32)
    value = np.take(colours, indices, axis=0)

    return value


def save_mixed_image(pred: np.ndarray, mask: np.ndarray, path: str):
    correct_1 = np.where(mask == 1, 1, 0) * np.where(pred == 1, 1, 0)
    correct_2 = np.where(mask == 2, 1, 0) * np.where(pred == 2, 1, 0)
    correct = np.where((correct_1 + correct_2) != 0, 1, 0)
    incorrect_1 = np.where(mask == 1, 1, 0) * np.where(pred != 1, 1, 0)
    incorrect_2 = np.where(mask == 2, 1, 0) * np.where(pred != 2, 1, 0)
    incorrect =  np.where((incorrect_1 + incorrect_2) != 0, 2, 0)
    mixed = correct + incorrect

    final = colourize(mixed.astype(np.float32), cmap='viridis')
    # Flip vertically as for some reason, saving PNG / JPG seems to save different to fits

    save_image(np.flip(final, 0), path, "PNG")


def mask_check(model, device, save_path, valid_set, save=False):
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

            if save:
                mid = prediction.detach().cpu().squeeze()
                mid = F.one_hot(mid.argmax(dim=0), 3).permute(3, 0, 1, 2)
                mid = np.argmax(mid, axis=0)
                pred = mid.amax(dim=0).numpy()
                mask = mask_image.amax(axis=0).cpu().squeeze().numpy()
                save_mixed_image(pred, mask, "./eval_" + str(idx-1) + ".png")  

            prediction = finalise_result(prediction)
            # Now compare both the prediction and the mask
            score = compare_masks(mask_image.numpy(), prediction)
            scores.append((score, source_path))

    visualise_scores(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sea-elegance run")
    parser.add_argument("--load", default=".", help="Path to our model dir.")
    parser.add_argument("--data", default=".", help="Path to our dataset dir.")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training."
    )
    parser.add_argument(
        "--save", action="store_true", default=False, help="Save the predicted images."
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

        mask_check(model, device, args.data, valid_paths, args.save)

    else:
        print("--load must point to a run directory. --data must point to the dataset")
        sys.exit(0)

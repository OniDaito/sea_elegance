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
import scipy.stats
import matplotlib
import matplotlib.cm
from astropy.io import fits
import torch.nn.functional as F
from util.loadsave import load_checkpoint, load_model
from util.image import load_fits, save_image, save_fits, finalise_result


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
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

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
    correct_1 = np.where(mask == 1, 5, 0) * np.where(pred == 1, 1, 0)
    correct_2 = np.where(mask == 2, 10, 0) * np.where(pred == 2, 1, 0)
    incorrect_1 = np.where(mask == 1, 30, 0) * np.where(pred != 1, 1, 0)
    incorrect_2 = np.where(mask == 2, 50, 0) * np.where(pred != 2, 2, 0)
    mixed = correct_1 + correct_2 + incorrect_1 + incorrect_2

    final = colourize(mixed.astype(np.float32), cmap='viridis')
    # Flip vertically as for some reason, saving PNG / JPG seems to save different to fits

    save_image(np.flip(final, 0), path, "PNG")


def compare_masks(original: np.ndarray, predicted: np.ndarray):
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


def mask_check(model, device, save_path, valid_set, save=False):
    """
    Test our model 

    Parameters
    ----------
    model : NetU
        The saved, trained model

    device : torch.Device
        The device to run on (cpu or cuda)

    save_path : str
        Path to the saved run

    valid_set : dictionary
        Two lists of input images and target pairs. We will check
        them over and produce some stats

    Returns
    -------
    None
    """

    # Need to call model.eval() to set certain layers to eval mode. Ones
    # like dropout and what not

    with open("eval_log.csv", "w") as w:
        w.write("idx,Overlay,Overlap,Dice,Jaccard,Dice1,Jacc1,tp1,tn1,fp1,fn1,Dice2,Jacc2,tp2,tn2,fp2,fn2\n")

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
                save_fits(mid, "./eval_" + str(idx-1) + ".fits")
                pred = mid.amax(dim=0).numpy()
                mask = mask_image.amax(axis=0).cpu().squeeze().numpy()
                save_mixed_image(pred, mask, "./eval_" + str(idx-1) + ".png")

            prediction = finalise_result(prediction)
            # Now compare both the prediction and the mask
            scores = compare_masks(mask_image.numpy(), prediction)

            with open("eval_log.csv", "a") as w:
                w.write(str(idx) + "," + scores + "\n")


def fluoro_check(model, device, valid_set, save_path, data_dir):
    """
    Test our model by generating the masks for the valid set then
    using these masks to count the flouresence scores. Compare these
    scores to the ones from QueeLim's real data.

    Parameters
    ----------
    model : NetU
        The saved, trained model

    device : torch.Device
        The device to run on (cpu or cuda)

    valid_set : dictionary
        Two lists of input images and target pairs. We will check
        them over and produce some stats

    save_path : str
        Path to the saved run

    data_dir : str
        Path to the dataset

    Returns
    -------
    None
    """

    # Need to call model.eval() to set certain layers to eval mode. Ones
    # like dropout and what not
    scores_original = pd.read_csv(data_dir + "/fluoro.csv", names=["id", "asi1", "asi2", "asj1", "asj2", "file"],
                                  converters={'file': strip})
    final_scores = []

    for idx in range(1, int((valid_set.size) / 2)):
        source_path = data_dir + "/" + valid_set.iloc[idx, 0]
        mask_path = data_dir + "/" + valid_set.iloc[idx, 1]

        tokens = valid_set.iloc[idx, 0].split("_")
        file_id = int(tokens[0])
        asi1 = scores_original.iloc[file_id - 1, 1]
        asi2 = scores_original.iloc[file_id - 1, 2]
        asj1 = scores_original.iloc[file_id - 1, 3]
        asj2 = scores_original.iloc[file_id - 1, 4]

        og_asi = asi1 + asi2
        og_asj = asj1 + asj2

        source_image = load_image(source_path, normalise=True)
        source_base = load_image(source_path, normalise=False)
        mask_image = load_image(mask_path, normalise=False, dtype=torch.int64)

        with torch.no_grad():
            model.eval()
            im = source_image.unsqueeze(dim=0).unsqueeze(dim=0)
            im = im.to(device)
            prediction = model.forward(im)

            mid = prediction.detach().cpu().squeeze()
            mid = F.one_hot(mid.argmax(dim=0), 3)
            # Class 1
            c1 = mid[:, :, :, 1].squeeze()
            c1_count = float(torch.sum(c1 * source_base))

            # Class 2
            c2 = mid[:, :, :, 2].squeeze()
            c2_count = float(torch.sum(c2 * source_base))

            # Now use the OG masks

            # Class 1
            mid = F.one_hot(mask_image, 3)
            c1 = mid[:, :, :, 1].squeeze()
            m1_count = float(torch.sum(c1 * source_base))

            # Class 2
            c2 = mid[:, :, :, 2].squeeze()
            m2_count = float(torch.sum(c2 * source_base))

            final_scores.append(
                ((og_asi, og_asj), (c1_count, c2_count), (m1_count, m2_count)))

    with open("eval_fluro.txt", "w") as w:
        w.write(str(final_scores) + "\n")

    asi_x = [x[1][0] for x in final_scores]
    asi_y = [x[2][0] for x in final_scores]

    asj_x = [x[1][1] for x in final_scores]
    asj_y = [x[2][1] for x in final_scores]

    asi_r = np.corrcoef(asi_x, asi_y)
    asj_r = np.corrcoef(asj_x, asj_y)

    with open("eval_fluro.txt", "a") as w:
        w.write("Correlations " + str(asi_r) + ", " + str(asj_r) + "\n")
        w.write("ASI\n")
        w.write("Spearman " + str(scipy.stats.spearmanr(asi_x, asi_y)) + "\n")
        w.write("Kendall " + str(scipy.stats.kendalltau(asi_x, asi_y)) + "\n")

        w.write("ASJ\n")
        w.write("Spearman " + str(scipy.stats.spearmanr(asj_x, asj_y)) + "\n")
        w.write("Kendall " + str(scipy.stats.kendalltau(asj_x, asj_y)) + "\n")

    og_asi = [x[0][0] for x in final_scores]
    og_asj = [x[0][1] for x in final_scores]

    with open("eval_fluro.txt", "a") as w:
        w.write("ASI with OG\n")
        w.write("Spearman " + str(scipy.stats.spearmanr(asi_x, og_asi)) + "\n")
        w.write("Kendall " + str(scipy.stats.kendalltau(asi_x, og_asi)) + "\n")

        w.write("ASJ with OG\n")
        w.write("Spearman " + str(scipy.stats.spearmanr(asj_x, og_asj)) + "\n")
        w.write("Kendall " + str(scipy.stats.kendalltau(asj_x, og_asj)) + "\n")


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
        fluoro_check(model, device, valid_paths, args.load, args.data)

    else:
        print("--load must point to a run directory. --data must point to the dataset")
        sys.exit(0)

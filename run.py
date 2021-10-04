""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

run.py - pass in a worm image, get a classification out.

To load a trained network:
  python run.py --load checkpoint.pth.tar --image <X> --points <Y>

  Load a checkpoint and an image (FITS) and output an
  image and some angles.

  For example:

  python run.py --load ../runs/2021_05_06_bl_0 --image renderer.fits

"""

import torch
import argparse
import sys
import os
from util.loadsave import load_checkpoint, load_model
from util.image import load_fits, save_fits, save_image


def image_test(model, device, input_image):
    """Test our model by loading an image and seeing how well we
    can match it. We might need to duplicate to match the batch size.
    """
 
    # Need to call model.eval() to set certain layers to eval mode. Ones
    # like dropout and what not
    with torch.no_grad():
        model.eval()
        im = input_image.unsqueeze(dim=0).unsqueeze(dim=0)
        im = im.to(device)
        x = model.forward(im)
        x = torch.squeeze(x)
        im = torch.squeeze(im)
        save_image(x, name="guess.jpg")
        save_fits(x, name="guess.fits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sea-elegance run")
    parser.add_argument("--load", default=".", help="Path to our model dir.")
    parser.add_argument(
        "--image", default="input.fits", help="An input image in FITS format"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.load and os.path.isfile(args.load + "/checkpoint.pth.tar"):
        # (savedir, savename) = os.path.split(args.load)
        # print(savedir, savename)
        model = load_model(args.load + "/model.tar")
        (model, _, _, _, _, prev_args) = load_checkpoint(
            model, args.load, "checkpoint.pth.tar", device
        )
        model = model.to(device)
        model.eval()
    else:
        print("--load must point to a run directory.")
        sys.exit(0)

    # Potentially load a different set of points

    if os.path.isfile(args.image):
        input_image = load_fits(args.image, dtype=torch.float16)
        image_test(model, device, input_image)
    else:
        print("--image must point to a valid fits file.")
        sys.exit(0)

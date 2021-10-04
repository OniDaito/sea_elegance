""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

run.py - an attempt to find the 3D shape from an image.

To load a trained network:
  python run.py --load checkpoint.pth.tar --image <X> --points <Y>

  Load a checkpoint and an image (FITS) and output an
  image and some angles.

  For example:

  python run.py --load ../runs/2021_05_06_bl_0 --image renderer.fits --points ../runs/2021_05_06_bl_0/last.ply

"""
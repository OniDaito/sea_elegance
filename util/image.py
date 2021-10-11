"""   # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/           # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/           # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_image.py - save out our images & load images
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy import ndimage as nd


def save_image(img_tensor, name="ten.jpg"):
    """
    Save a particular tensor to an image. We add a normalisation here
    to make sure it falls in range.

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    None
    """
    if hasattr(img_tensor, "detach"):
        img_tensor = img_tensor.detach().cpu()
        mm = torch.max(img_tensor)
        img_tensor = img_tensor / mm
        img = Image.fromarray(np.uint8(img_tensor.numpy() * 255))
        img.save(name, "JPEG")
    else:
        # Assume a numpy array
        mm = np.max(img_tensor)
        img_tensor = img_tensor / mm
        img = Image.fromarray(np.uint8(img_tensor * 255))
        img.save(name, "JPEG")


def save_fits(img_tensor, name="ten.fits", dtype=np.float32):
    """
    Save a particular tensor to an image. We add a normalisation here
    to make sure it falls in range. This version saves as a

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    self
    """

    from astropy.io import fits

    if hasattr(img_tensor, "detach"):
        img_tensor = np.flipud(img_tensor.detach().cpu().numpy())
    img_tensor = img_tensor.astype(dtype)
    hdu = fits.PrimaryHDU(img_tensor)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name)


def load_fits(path, dtype=torch.float32):
    """
    Load a FITS image file, converting it to a Tensor
    Parameters
    ----------
    path : str
        The path to the FITS file.

    Returns
    -------
    torch.Tensor
    """
    from astropy.io import fits

    hdul = fits.open(path)
    # Intermediate conversion - could introduce badness
    data = np.array(hdul[0].data).astype(np.float32)
    return torch.tensor(data, dtype=dtype)


def load_image(path):
    """
    Load a standard image like a png or jpg using PIL/Pillow,
    assuming the image has values 0-254. Convert to a
    normalised tensor.

    Parameters
    ----------
    path : str
        The path to the image

    Returns
    -------
    torch.Tensor
    """
    im = Image.open(path)
    nm = np.asarray(im, dtype=np.float32)
    nm = nm / 255.0
    return torch.tensor(nm, dtype=torch.float32)


def resize_3d(image, zoom=1.0) -> torch.Tensor:
    """
    Resize our 3D image

    Parameters
    ----------
    image : torch.tensor
        The input image we want to resize

    zoom : float
        The zoom factor for our new image

    Returns
    -------
    torch.Tensor
    """

    target = nd.interpolation.zoom(image.cpu().numpy(), zoom=0.5)
    return torch.tensor(target)


def reduce_source(image, axis=1) -> np.ndarray:
    """
    Reduce the 3D image batch to a 2D single image 

    Parameters
    ----------
    image : torch.tensor
        The batch of 3D input images

    axis : int
        The axis to squish down - default 1.

    Returns
    -------
    torch.Tensor
    """

    m = torch.max(image[0]).item()
    final = image[0].amax(axis=axis).cpu().numpy()
    return np.array(final / m * 255).astype(np.uint8)


def reduce_mask(image, axis=0) -> np.ndarray:
    """
    Reduce the 3D image batch input mask to a 2D single image 

    Parameters
    ----------
    image : torch.tensor
        The batch of 3D input images

    axis : int
        The axis to squish down - default 0.

    Returns
    -------
    torch.Tensor
    """
    final = image[0].amax(axis=axis).cpu().unsqueeze(dim=0).numpy().astype(np.float32)
    return np.array(final / 4 * 255).astype(np.uint8)


def reduce_result(image, axis=0) -> np.ndarray:
    """
    Reduce the 3D image mask result batch to a 2D single image 

    Parameters
    ----------
    image : torch.tensor
        The batch of 3D input images

    axis : int
        The axis to squish down - default 1.

    Returns
    -------
    torch.Tensor
    """
    first = image[0].detach().cpu().squeeze()
    mid = first
    mid = F.one_hot(mid.argmax(dim=0), 5).permute(3, 0, 1, 2)
    mid = np.argmax(mid, axis=0)* 255 / mid.shape[0]
    mid = mid.amax(dim = axis)
    return mid.numpy().astype(np.uint8)


""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

image.py - tests for images

"""

import unittest
import train_unet
import torch
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from util.image import reduce_result


class Image(unittest.TestCase):
    def test_res(self):
        chess_board = []
        for y in range(150):
            row = []
            a = y // 10
            col0 = a % 2
            for x in range(320):
                d = x // 10
                col1 = d % 2
                final = 0
                if not (col0 == col1):
                    final = 1
                row.append(final)
            chess_board.append(row)
    
        stack = []
        for i in range(26):
            stack.append(chess_board)            

        final = torch.tensor(stack)
        final = final.unsqueeze(dim=0)

        image_top = reduce_result(final)
        image_top = image_top.squeeze()

        image_side = reduce_result(final, axis=1)
        image_side = image_side.squeeze()

        f, axarr = plt.subplots(2, 1)
        axarr[0].imshow(image_top)
        axarr[1].imshow(image_side)
        plt.show()

    def test_permute(self):
        img_path = "./test/images/mask.fits"
        with fits.open(img_path) as w:
            hdul = w[0].data.byteswap().newbyteorder()
            source_image = np.array(hdul).astype("int8")
            source = torch.tensor(source_image, dtype=torch.long)

            #reduced = np.max(source_image.astype(float), axis=1)
            #c = plt.imshow(reduced)
            #plt.show()

    def test_output(self):
        img_path = "./test/prediction.npy"
        with open(img_path, 'rb') as f:
            prediction = np.load(f).astype("float32")
            batch = torch.tensor(prediction).unsqueeze(dim=0) 
            reduced = reduce_result(batch)
            from PIL import Image
            im = Image.fromarray(reduced).convert('RGB')
            im.save("test_output.png")


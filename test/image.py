
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

        image_top = train_unet.convert_result(final)
        image_top = image_top.squeeze()

        image_side = train_unet.convert_result(final, axis=1)
        image_side = image_side.squeeze()

        f, axarr = plt.subplots(2, 1)
        axarr[0].imshow(image_top)
        axarr[1].imshow(image_side)
        plt.show()

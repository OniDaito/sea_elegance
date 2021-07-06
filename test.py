""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

test.py - our test functions using python's unittest


"""

import unittest
from test.data import Data

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Data(img_dir="test_loader"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
    unittest.main()

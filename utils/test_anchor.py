from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np 

from anchor import Anchor

class Anchor_Tests(unittest.TestCase):
    def test_generate(self):
        anchor = Anchor()
        base_anchors = anchor.generate()
        self.assertEqual(base_anchors.shape[0],9)



if __name__ == '__main__':
    unittest.main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

from labelrecord import LabelRecord

LABEL_FILE = '../input/stage_1_train_labels.csv'

class LabelRecord_Tests(unittest.TestCase):
    def test_load(self):
        exists = True
        try:
            f = open(LABEL_FILE)
            f.close()
        except IOError:
            exists = False
        if exists:
            lr = LabelRecord()
            records = lr.load(LABEL_FILE)
            self.assertTrue(len(records) > 0,'Could not load label file')
        else:
            self.skipTest("Label file %s not accessible, can't proceed with test" % LABEL_FILE)

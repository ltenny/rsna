from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

from labelrecord import LabelRecord
from box import Box

LABEL_FILE = '../input/stage_1_train_labels.csv'

class Box_Tests(unittest.TestCase):
    def test_get_all_bounding_boxes(self):
        exists = True
        try:
            f = open(LABEL_FILE)
            f.close()
        except IOError:
            exists = False
        if exists:
            lr = LabelRecord()
            records = lr.load(LABEL_FILE)
            all = Box.get_all_bounding_boxes(records)
            self.assertTrue(len(all) > 0,'No bounding boxes!')

        else:
            self.skipTest("Label file %s not accessible, can't proceed with test" % LABEL_FILE)

    def test_clip_boxes(self):
        exists = True
        try:
            f = open(LABEL_FILE)
            f.close()
        except IOError:
            exists = False
        if exists:
            lr = LabelRecord()
            records = lr.load(LABEL_FILE)
            all = Box.get_all_bounding_boxes(records)
            clipped = Box.clip_boxes(all, 500, 500)
            self.assertTrue(len(all) > 0,'No bounding boxes!')

        else:
            self.skipTest("Label file %s not accessible, can't proceed with test" % LABEL_FILE)

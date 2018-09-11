# 
# This simple app extracts the training images from the .dcm files. See the README.md
# for more details.
#

import tensorflow as tf
import numpy as np
import sys
from utils.cxrimage import CXRImage
from utils.labelrecord import LabelRecord
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('examples_dir','examples',"""Directory for the cxr example images""")
tf.app.flags.DEFINE_string('originals_dir','originals',"""Directory for the original cxr images""")
tf.app.flags.DEFINE_string('positives_dir','positives',"""Directory for the positive image regions""")
tf.app.flags.DEFINE_string('negatives_dir','negatives',"""Directory for the negative image regions""")
tf.app.flags.DEFINE_string('label_file','input/stage_1_train_labels.csv',"""Path to labels .csv file""")
tf.app.flags.DEFINE_string('image_path','input/stage_1_train_images',"""Root directory for the .dcm files containing the train CXRs""")

def _init_output_directories():
    if tf.gfile.Exists(FLAGS.positives_dir):
        tf.gfile.DeleteRecursively(FLAGS.positives_dir)
    tf.gfile.MakeDirs(FLAGS.positives_dir)

    if tf.gfile.Exists(FLAGS.negatives_dir):
        tf.gfile.DeleteRecursively(FLAGS.negatives_dir)
    tf.gfile.MakeDirs(FLAGS.negatives_dir)

    if tf.gfile.Exists(FLAGS.originals_dir):
        tf.gfile.DeleteRecursively(FLAGS.originals_dir)
    tf.gfile.MakeDirs(FLAGS.originals_dir)

    if tf.gfile.Exists(FLAGS.examples_dir):
        tf.gfile.DeleteRecursively(FLAGS.examples_dir)
    tf.gfile.MakeDirs(FLAGS.examples_dir)


# returns collection of all bounding boxes
def _get_bounding_boxes(records):
    allboxes = np.array([],np.int32)
    for _,v in records.items():
        if v.hasBoundingBox:
            allboxes = np.append(allboxes, v.boundingBoxes)
    return allboxes

# main entry point
def main(argv=None):
    _init_output_directories()

    print('loading labels from %s' % FLAGS.label_file)
    lr = LabelRecord()
    label_records = lr.load(FLAGS.label_file)

    all_bounding_boxes = _get_bounding_boxes(label_records)

    for (_,v) in label_records.items():
        print('processing %s' % v.filename)
        image = CXRImage.get_image_data(v.filename, FLAGS.image_path)
        basefilename = os.path.splitext(v.filename)[0]
        if v.hasBoundingBox:
            for i in range(0,v.boundingBoxes.shape[0],4):
                box = v.boundingBoxes[i:i+4]
                CXRImage.write_image(CXRImage.extract_image(image, box),FLAGS.positives_dir)
            CXRImage.write_image(image, FLAGS.examples_dir, "%s.jpg" % basefilename)
        else:
            i = np.int32((np.random.randint(0, np.max(all_bounding_boxes.shape[0]-4,0)))/ 4) * 4
            box = all_bounding_boxes[i:i+4]
            CXRImage.write_image(CXRImage.extract_image(image, box), FLAGS.negatives_dir)

        CXRImage.write_image_with_bounding_boxes(image, FLAGS.originals_dir, "%s.jpg" % basefilename, v.boundingBoxes)

if __name__ == '__main__':
    tf.app.run()
    
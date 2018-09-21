# 
# This simple app extracts the training images from the .dcm files. See the README.md
# for more details.
#

import tensorflow as tf
import numpy as np
import sys
from utils.cxrimage import CXRImage
from utils.labelrecord import LabelRecord
from utils.record import Record
from utils.box import Box
from utils.progress import Progress
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('do_full_prepare', False,'Do full prepare, first time requires full prepare')
tf.app.flags.DEFINE_string('examples_dir','examples','Directory for the cxr example images')
tf.app.flags.DEFINE_string('example_file','examples.tfrecord','Filename for the example data')
tf.app.flags.DEFINE_string('pre_train_file','pre_train.tfrecord','TFRecord containing pre-training features')
tf.app.flags.DEFINE_string('originals_dir','originals','Directory for the original cxr images')
tf.app.flags.DEFINE_string('positives_dir','positives','Directory for the positive image regions')
tf.app.flags.DEFINE_string('negatives_dir','negatives','Directory for the negative image regions')
tf.app.flags.DEFINE_string('label_file','input/stage_1_train_labels.csv','Path to labels .csv file')
tf.app.flags.DEFINE_string('image_path','input/stage_1_train_images','Root directory for the .dcm files containing the train CXRs')
tf.app.flags.DEFINE_boolean('grayscale',True,'If True, the TFRecord file is created with 1 channel, grayscale')

def _init_output_directories():
    if FLAGS.do_full_prepare:
        if tf.gfile.Exists(FLAGS.positives_dir):
            print('Deleting %s...' % FLAGS.positives_dir)
            tf.gfile.DeleteRecursively(FLAGS.positives_dir)
        tf.gfile.MakeDirs(FLAGS.positives_dir)

        if tf.gfile.Exists(FLAGS.negatives_dir):
            print('Deleting %s...' % FLAGS.negatives_dir)
            tf.gfile.DeleteRecursively(FLAGS.negatives_dir)
        tf.gfile.MakeDirs(FLAGS.negatives_dir)

        if tf.gfile.Exists(FLAGS.originals_dir):
            print('Deleting %s...' % FLAGS.originals_dir)
            tf.gfile.DeleteRecursively(FLAGS.originals_dir)
        tf.gfile.MakeDirs(FLAGS.originals_dir)

        if tf.gfile.Exists(FLAGS.examples_dir):
            print('Deleting %s...' % FLAGS.examples_dir)
            tf.gfile.DeleteRecursively(FLAGS.examples_dir)
        tf.gfile.MakeDirs(FLAGS.examples_dir)


# main entry point
def main(argv=None):
    _init_output_directories()

    # step 1: create the negatives and positives directory
    if FLAGS.do_full_prepare:
        print('Loading labels from %s' % FLAGS.label_file)
        lr = LabelRecord()
        label_records = lr.load(FLAGS.label_file)
        all_bounding_boxes = Box.get_all_bounding_boxes(label_records)
        counter = 0
        
        # fill examples, originals, negatives, and positives directories
        print('Processing images...')
        for (_,v) in label_records.items():
            Progress.show_progress(counter)
            image = CXRImage.get_image_data(v.filename, FLAGS.image_path)
            basefilename = os.path.splitext(v.filename)[0]
            if v.hasBoundingBox:
                for i in range(0,v.boundingBoxes.shape[0]):
                    box = v.boundingBoxes[i,:]
                    CXRImage.extract_center_and_write(image,box,1024,1024,FLAGS.positives_dir)
                CXRImage.write_image(image, FLAGS.examples_dir, "%s.jpg" % basefilename)
            else:
                i = np.int32(np.random.randint(0, all_bounding_boxes.shape[0] - 1))
                box = all_bounding_boxes[i,:]
                CXRImage.extract_center_and_write(image,box,1024,1024,FLAGS.negatives_dir)

            CXRImage.write_image_with_bounding_boxes(image, FLAGS.originals_dir, "%s.jpg" % basefilename, v.boundingBoxes)
            counter = counter + 1
    
    # step 2: create the pre-training features by combining negatives and positives into pre_train.tfrecord
    print('\nCreating pre-train file...')
    rec = Record(1024,1024,1 if FLAGS.grayscale else 3)
    total = rec.create_pre_train_file(FLAGS.positives_dir, FLAGS.negatives_dir, FLAGS.pre_train_file)
    print('\n%d files combined in %s' % (total, FLAGS.pre_train_file))

if __name__ == '__main__':
    tf.app.run()
    
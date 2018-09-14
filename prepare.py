# 
# This simple app extracts the training images from the .dcm files. See the README.md
# for more details.
#

import tensorflow as tf
import numpy as np
import sys
from utils.cxrimage import CXRImage
from utils.labelrecord import LabelRecord
from utils.box import Box
import os
import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('examples_dir','examples',"""Directory for the cxr example images""")
tf.app.flags.DEFINE_string('originals_dir','originals',"""Directory for the original cxr images""")
tf.app.flags.DEFINE_string('positives_dir','positives',"""Directory for the positive image regions""")
tf.app.flags.DEFINE_string('negatives_dir','negatives',"""Directory for the negative image regions""")
tf.app.flags.DEFINE_string('label_file','input/stage_1_train_labels.csv',"""Path to labels .csv file""")
tf.app.flags.DEFINE_string('image_path','input/stage_1_train_images',"""Root directory for the .dcm files containing the train CXRs""")

def _init_output_directories():
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

def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))    

def _create_tf_records(records):
    cnt = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.examples_dir,'examples.txt'))
    for (_,v) in records.items():
        if v.hasBoundingBox:
            image = CXRImage.get_image_data(v.filename, FLAGS.image_path)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_list_feature(image.flatten().tobytes()),
                'image/width': _int64_feature(image.shape[0]),
                'image/height': _int64_feature(image.shape[1]),
                'image/channels': _int64_feature(image.shape[2]),
                'image/format': _bytes_list_feature('jpg'.encode('utf8')),
                'filename' : _bytes_list_feature((os.path.splitext(v.filename)[0]).encode('utf8')),
                'ground_true_boxes': _int64_list_feature(v.boundingBoxes.flatten())
            }))
            cnt = cnt + 1
            print('writing record %d    ' % cnt, end="\r")
            writer.write(example.SerializeToString())
    writer.close()

# main entry point
def main(argv=None):
    _init_output_directories()

    print('loading labels from %s' % FLAGS.label_file)
    lr = LabelRecord()
    label_records = lr.load(FLAGS.label_file)

    all_bounding_boxes = Box.get_all_bounding_boxes(label_records)
    _create_tf_records(label_records)

    for (_,v) in label_records.items():
        print('processing %s' % v.filename)
        image = CXRImage.get_image_data(v.filename, FLAGS.image_path)
        basefilename = os.path.splitext(v.filename)[0]
        if v.hasBoundingBox:
            for i in range(0,v.boundingBoxes.shape[0]):
                box = v.boundingBoxes[i,:]
                CXRImage.write_image(CXRImage.extract_image(image, box),FLAGS.positives_dir)
            #CXRImage.write_image(image, FLAGS.examples_dir, "%s.jpg" % basefilename)
        else:
            i = np.int32(np.random.randint(0, all_bounding_boxes.shape[0] - 1))
            box = all_bounding_boxes[i,:]
            CXRImage.write_image(CXRImage.extract_image(image, box), FLAGS.negatives_dir)

        CXRImage.write_image_with_bounding_boxes(image, FLAGS.originals_dir, "%s.jpg" % basefilename, v.boundingBoxes)

if __name__ == '__main__':
    tf.app.run()
    
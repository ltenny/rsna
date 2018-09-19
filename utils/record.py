import tensorflow as tf
from utils.cxrimage import CXRImage
from utils.labelrecord import LabelRecord
from utils.box import Box
from utils.progress import Progress
import os
import glob
from PIL import Image
import numpy as np 

# class to manage data records
class Record(object):
    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    @staticmethod
    def make_labeled_example(image, label):
        image_data = np.array(image)
        return tf.train.Example(features=tf.train.Features(feature={
            'image': Record.bytes_feature(image_data.flatten().tobytes()),
            'image/width': Record.int64_feature(image_data.shape[0]),
            'image/height': Record.int64_feature(image_data.shape[1]),
            'image/channels': Record.int64_feature(image_data.shape[2]),
            'label': Record.int64_feature(label)}))

    # for pre-training, create a TFRecord file containing all positive and negative examples
    @staticmethod
    def create_pre_train_file(positives, negatives, filename):
        total = 0
        writer = tf.python_io.TFRecordWriter(filename)
        for name in glob.glob(os.path.join(positives,'*.jpg')):
            image = Image.open(name)
            writer.write(Record.make_labeled_example(image,1).SerializeToString())
            total = total + 1
            Progress.show_progress(total)

        for name in glob.glob(os.path.join(negatives,'*.jpg')):
            image = Image.open(name)
            writer.write(Record.make_labeled_example(image,0).SerializeToString())
            total = total + 1
            Progress.show_progress(total)

        writer.close()
        return total

                    



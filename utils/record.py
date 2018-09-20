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

    # for pre-training, create a TFRecord file containing all positive and negative examples, random shuffle
    @staticmethod
    def create_pre_train_file(positives, negatives, filename):
        total = 0
        writer = tf.python_io.TFRecordWriter(filename)
        p =  [{'name': name, 'label': 1} for name in glob.glob(os.path.join(positives, '*.jpg'))]
        n =  [{'name': name, 'label': 0} for name in glob.glob(os.path.join(negatives, '*.jpg'))]
        recs = p + n

        np.random.shuffle(recs)
        for r in recs:
            image = Image.open(r['name'])
            writer.write(Record.make_labeled_example(image,r['label']).SerializeToString())
            total = total + 1
            Progress.show_progress(total)

        writer.close()
        return total

    # for pre-trainig, read from the TFRecord created by create_pre_train_file() above
    # NB: this must be called in context of a tf graph and filenames is [file1, file2,...]
    @staticmethod
    def read_pre_train_record(filenames):
        with tf.name_scope('pre_train_data_augmentation'):
            feature = {
                'image': tf.FixedLenFeature([], tf.string),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/channels': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64)
            }
            filename_queue = tf.train.string_input_producer(filenames)
            reader = tf.TFRecordReader()
            _, example = reader.read(filename_queue)
            features = tf.parse_single_example(example, features=feature)
            image = tf.decode_raw(features['image'], tf.uint8)
            label = tf.cast(features['label'], tf.int32)
            label = tf.cast(label,tf.float32)
            width = tf.cast(features['image/width'], tf.int32)
            height = tf.cast(features['image/height'], tf.int32)
            channels = tf.cast(features['image/channels'], tf.int32)

            label = tf.reshape(label,[])
            image = tf.reshape(image, [height, width, channels])
            float_image = tf.cast(image, tf.float32)

            # data augmentation - introduce random brightness and random contrast
            distorted_image = tf.image.random_brightness(float_image, max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

            standardized_image = tf.image.per_image_standardization(distorted_image)
            standardized_image.set_shape([1024,1024,3])
            return standardized_image, label

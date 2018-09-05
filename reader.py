import tensorflow as tf 
import numpy as np 
import os

IMAGE_SIZE = 1024

NUM_EXAMPLES_FOR_TRAIN = 25680  # for stage 1, might change in the future
NUM_EXAMPLES_FOR_TEST = 500     # check this

# returns ground truth boxes for the given file
# TODO: actually get the boxes if any
def get_boxes(filename):
    #for now
    return tf.convert_to_tensor(np.array[[1,1,1,1]], dtype=tf.int32)

# reads a single example and returns the image along with the ground truth boxes
def read_example(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    boxes = tf.py_func(get_boxes, [key], tf.TensorArray)
    return image, boxes


def create_examples_batch(image, gtboxes, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, boxes = tf.train.shuffle_batch(
            [image, gtboxes],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
    else:
        images, boxes = tf.train.batch(
            [image, gtboxes],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size)

    return images, tf.reshape(boxes, [batch_size])

# create a test batch
def inputs(test_dir, batch_size):
    filenames = tf.gfile.Glob(os.path.join(test_dir,'**/*.jpg'))
    num_examples_per_epoch = NUM_EXAMPLES_FOR_TEST

    filename_queue = tf.train.string_input_producer(filenames)
    with tf.name_scope('inputs'):
        image, boxes = read_example(filename_queue)
        float_image = tf.cast(image, tf.float32)
        standardized_image = tf.image.per_image_standardization(float_image)
        standardized_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        return create_examples_batch(standardized_image, boxes, min_queue_examples, batch_size, shuffle=True)

# for training, we do a bit of data augmentation by changing contrast and brightness, other kinds of augmentation
# would probably screw up our bounding boxes (e.g. flipping left/right, random cropping, etc.)
def distorted_inputs(train_dir, batch_size):
    filenames = tf.gfile.Glob(os.path.join(train_dir,'**/*.jpg'))
    num_examples_per_epoch = NUM_EXAMPLES_FOR_TRAIN

    filename_queue = tf.train.string_input_producer(filenames)
    with tf.name_scope('inputs'):
        image, boxes = read_example(filename_queue)
        float_image = tf.cast(image, tf.float32)
        distorted_image = tf.image.random_brightness(float_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        standardized_image = tf.image.per_image_standardization(distorted_image)
        standardized_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        return create_examples_batch(standardized_image, boxes, min_queue_examples, batch_size, shuffle=True)

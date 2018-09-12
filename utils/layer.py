from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Layer(object):

    @staticmethod
    def reshape(bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            to_caffe = tf.transpose(bottom, [0,3,1,2])
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            to_tf = tf.transposed(reshaped, [0,2,3,1])
            return to_tf 

        
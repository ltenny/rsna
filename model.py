from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorboard.plugins.beholder import BeholderHook

import re
import numpy as np 
from utils.anchor import Anchor
from utils.layer import Layer
from utils.record import Record
from datetime import datetime
import time

TOWER_NAME = 'tower'
class Model(object):
    def __init__(self, use_fp16=True, scales=(8,16,32), ratios=(0.5,1,2)):
        self.scales = scales
        self.use_fp16 = use_fp16
        self.ratios = ratios
        self.k_anchors = len(scales) * len(ratios)
        self.learning_rate = 0.001
        self.optimizer_decay = 0.5
        self.optimizer_momentum = 0.0
        self.pre_train_step = 0
        self.log_frequency = 100
        self.init_pre_train_complete = False
        self.pre_train_height = 256
        self.pre_train_width = 256

    def init_pre_train(self, batch_size=32, state_dir = 'pretrain_dir', filename='pre_train.tfrecord', max_steps = 500000, 
            threads = 16, examples_per_epoch = 28000, min_fraction = 0.4):
        self.pre_train_batch_size = batch_size
        self.pre_train_number_of_threads = threads
        self.pre_train_number_of_examples_per_epoch = examples_per_epoch
        self.pre_train_min_fraction_of_examples_in_queue = min_fraction
        self.pre_train_max_steps = max_steps
        self.pre_train_dir = state_dir
        self.pre_train_file = filename
        self.init_pre_train_complete = True


    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float16 if self.use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd=None):
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        dtype = tf.float16 if self.use_fp16 else tf.float32
#        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        var = self._variable_on_cpu(name, shape, initializer)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(self, x):
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _get_pretrain_op(self, loss, global_step):
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.optimizer_decay)
        grads = optimizer.compute_gradients(loss)
        return optimizer.apply_gradients(grads,global_step=global_step)

    def _pretrain_loss(self, pretrain, labels):
        #loss = tf.reduce_mean(tf.squared_difference(tf.reshape(pretrain,[-1]), labels))
        loss = tf.losses.log_loss(tf.reshape(labels,[64,1]), tf.reshape(pretrain,[64,1]),weights=2.0,epsilon=1e-4)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("positive_labels",tf.count_nonzero(labels))
        return loss

    def pre_train(self):
        if not self.init_pre_train_complete:
            print('init_pre_train() not called!')
            return

        print(' ')
        print('Pre-training with the following parameters:')
        print('\tState directory: %s' % self.pre_train_dir)
        print('\tTrainable variables are 16 bit? : %s' % ('Yes' if self.use_fp16 else 'No'))
        print('\tBatch size: %d' % self.pre_train_batch_size)
        print('\tTraining images are: [%d,%d,%d]' % (self.pre_train_height, self.pre_train_width, 1))
        print('\tLearning rate: %.3f' % self.learning_rate)
        print('\tRMSProp Optimizer decay: %.3f' % self.optimizer_decay)
        print('\tRMSProp Optimizer momentum: %.3f' % self.optimizer_momentum)
        print(' ')

        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            rec = Record(self.pre_train_height, self.pre_train_width, 1, self.use_fp16)
            with tf.device('/cpu:0'):
                image, label = rec.read_pre_train_record([self.pre_train_file])
                images, labels = tf.train.batch([image, label],
                    batch_size = self.pre_train_batch_size,
                    num_threads = self.pre_train_number_of_threads
                )
            
            _, pretrain = self.feature_network(images)
            loss = self._pretrain_loss(pretrain, labels)
            pretrain_op = self._get_pretrain_op(loss, global_step)

            batch_size = self.pre_train_batch_size
            logfreq = self.log_frequency
            #beholder_hook = BeholderHook(self.pre_train_dir)
            class _LoggerHook(tf.train.SessionRunHook):
                def begin(self):
                    self.pre_train_step = -1
                    self.start_time = time.time()
                
                def before_run(self, run_context):
                    self.pre_train_step += 1
                    return tf.train.SessionRunArgs(loss)

                def after_run(self, run_context, run_values):
                    if self.pre_train_step % logfreq == 0:
                        current_time = time.time()
                        duration = current_time - self.start_time
                        self.start_time = current_time

                        loss_value = run_values.results
                        examples_per_sec = logfreq * batch_size / duration
                        sec_per_batch = float(duration / logfreq)

                        format_str = ('%s: step %d, loss = %.3f  (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), self.pre_train_step, loss_value, examples_per_sec, sec_per_batch))
                
            with tf.train.MonitoredTrainingSession(
                checkpoint_dir = self.pre_train_dir,
                hooks = [tf.train.StopAtStepHook(last_step=self.pre_train_max_steps),
                            tf.train.NanTensorHook(loss),
                            _LoggerHook()],
                config = tf.ConfigProto(log_device_placement=False)) as mon_sess:

                while not mon_sess.should_stop():
                    mon_sess.run([pretrain_op])


    # roughly follow ConvNet layers from VGG-16 with changes from S. Ren et.al. (Faster R-CNN)
    # NB: the code is completely un-rolled to make it easy to understand and tweak
    def feature_network(self, images):
        with tf.variable_scope('conv1') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,1,64], stddev=5e-2)
            conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv1)

        with tf.variable_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,64,64], stddev=5e-2)
            conv = tf.nn.conv2d(conv1, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv2)

        max_pool1 = tf.nn.max_pool(conv2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool1')

        with tf.variable_scope('conv3') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,64,128], stddev=5e-2)
            conv = tf.nn.conv2d(max_pool1, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv3)

        with tf.variable_scope('conv4') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,128,128], stddev=5e-2)
            conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv4)

        max_pool2 = tf.nn.max_pool(conv4,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool2')

        with tf.variable_scope('conv5') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,128,256], stddev=5e-2)
            conv = tf.nn.conv2d(max_pool2, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            residual_1 = pre_activation
            conv5 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv5)

        with tf.variable_scope('conv6') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,256,256], stddev=5e-2)
            conv = tf.nn.conv2d(conv5, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv6 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv6)

        with tf.variable_scope('conv7') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,256,256], stddev=5e-2)
            conv = tf.nn.conv2d(conv6, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)  +  residual_1
            conv7 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv7)

        max_pool3 = tf.nn.max_pool(conv7,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool3')

        with tf.variable_scope('conv8') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,256,512], stddev=5e-2)
            conv = tf.nn.conv2d(max_pool3, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            residual_2 = pre_activation
            conv8 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv8)

        with tf.variable_scope('conv9') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,512,512], stddev=5e-2)
            conv = tf.nn.conv2d(conv8, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv9 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv9)

        with tf.variable_scope('conv10') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,512,512], stddev=5e-2)
            conv = tf.nn.conv2d(conv9, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases) + residual_2
            conv10 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv10)

        max_pool4 = tf.nn.max_pool(conv10,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool4')

        with tf.variable_scope('conv11') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,512,512], stddev=5e-2)
            conv = tf.nn.conv2d(max_pool4, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv11 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv11)

        with tf.variable_scope('conv12') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,512,512], stddev=5e-2)
            conv = tf.nn.conv2d(conv11, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv12 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv12)

        with tf.variable_scope('conv13') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,512,512], stddev=5e-2)
            conv = tf.nn.conv2d(conv12, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biasas', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases) + max_pool4
            conv13 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv13)
        
        # the following layers are used only for pre-training
        max_pool5 = tf.nn.max_pool(conv13, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool5')

        with tf.variable_scope('fc1') as scope:
            reshape = tf.reshape(max_pool5, [images.get_shape().as_list()[0], -1])
            dim = reshape.get_shape()[1].value
            weights = self._variable_with_weight_decay('weights', shape=[dim,384], stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            self._activation_summary(fc1)

        with tf.variable_scope('fc2') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
            self._activation_summary(fc2)

        with tf.variable_scope('fc3') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[192,1], stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
            pretrain = tf.nn.relu(tf.matmul(fc2, weights) + biases, name=scope.name)
            self._activation_summary(pretrain)

        return conv13, pretrain


    # builds the RPN
    def region_proposal_network(self, images, features):

        # with all 13 convolutional layers VGG-16 complete, now we take Ren's approach for the RPN 
        with tf.variable_scope('ren_conv') as scope:
            # build the anchors for the image
            height = tf.to_int32(tf.ceil(images.shape[1] / np.float32(16)))
            width = tf.to_int32(tf.ceil(images.shape[2] / np.float32(16)))
            anchors, anchors_length = Anchor.generate_anchors_initial(height, width, 16)
            anchors.set_shape([None, 4])
            anchors_length.set_shape([])

            # build the 3x3 convnet
            kernel = self._variable_with_weight_decay('weights', shape=[3,3,512,512], stddev=5e-2)
            conv = tf.nn.conv2d(features, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biases',[512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            ren_conv = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(ren_conv)

        # box-classification layer
        with tf.variable_scope('rpn_cls') as scope:
            channels = self.k_anchors * 2
            kernel = self._variable_with_weight_decay('weights', shape=[1,1,512,channels], stddev=5e-2)
            conv = tf.nn.conv2d(ren_conv, kernel, [1,1,1,1], padding='VALID')
            biases = self._variable_on_cpu('biases',[channels], tf.constant_initializer(0.0))
            rpn_cls_score = tf.nn.bias_add(conv, biases)
            rpn_cls_score_reshape = Layer.reshape(rpn_cls_score, 2, name='rpn_cls_score_reshape')
            
            input_shape = tf.shape(rpn_cls_score_reshape)
            reshaped = tf.reshape(rpn_cls_score_reshape, [-1, input_shape[-1]])
            score = tf.nn.softmax(reshaped)
            rpn_cls_prob_reshape = tf.reshape(score, input_shape)
            
            rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape,[-1,2]), axis=1)
            rpn_cls_prob = Layer.reshape(rpn_cls_prob_reshape, channels, name='rpn_cls_prob')
        
        # box-prediction layer
        with tf.variable_scope('rpn_bbox_pred') as scope:
            channels = self.k_anchors * 4
            kernel = self._variable_with_weight_decay('weights', shape=[1,1,512,channels], stddev=5e-2)
            conv = tf.nn.conv2d(ren_conv, kernel,[1,1,1,1], padding='VALID')
            biases = self._variable_on_cpu('biases',[channels], tf.constant_initializer(0,0))
            rpn_bbox_pred = tf.nn.bias_add(conv,biases)

        # TODO: complete this

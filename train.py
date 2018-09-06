from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import tensorflow as tf 
import time
import reader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_state_dir','train_state',"""Directory for the training state info""")
tf.app.flags.DEFINE_string('train_dir', 'Prepare/examples',"""Directory with raw cxr .jpg images""")
tf.app.flags.DEFINE_string('labelfile','input/stage_1_class_info.csv',"""Stage 1 training labels""")
tf.app.flags.DEFINE_integer('max_steps',500000,"""Max steps in training""")
tf.app.flags.DEFINE_integer('batch_size',32,"""Batch size""")
tf.app.flags.DEFINE_integer('log_frequency',10,"""How often to echo to the console""")
tf.app.flags.DEFINE_boolean('log_device_placement',False,"""Log device placement""")

def train():
    print("training...")
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.device('/cpu:0'):
            images, boxes = reader.distorted_inputs(FLAGS.train_dir,FLAGS.batch_size)

        logits = None   #TODO
        loss = None     #TODO  
        train_op = None #TODO

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    
                    #loss_value = run_values.results
                    loss_values = .1
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    format_str = ('%s: step: %d, loss = %.2f (%.1f examples/sec, %.3f sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

            with tf.train.MonitoredTrainingSession(
                checkpoint_dir = FLAGS.train_state_dir,
                hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                        tf.train.NanTensorHook(loss), _LoggerHook()],
                config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:

                    while not mon_sess.should_stop():
                        mon_sess.run(train_op)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_state_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_state_dir)
    tf.gfile.MakeDirs(FLAGS.train_state_dir)
    train()

if __name__ == '__main__':
    tf.app.run()

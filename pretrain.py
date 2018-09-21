import tensorflow as tf
import os
from model import Model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use tf.float16 vs tf.float32')
tf.app.flags.DEFINE_integer('log_frequency',100,'Log freqency')
tf.app.flags.DEFINE_integer('batch_size',16, 'Batch size for pre-training')
tf.app.flags.DEFINE_string('train_file','pre_train.tfrecord','TFRecord containing pre-training features')
tf.app.flags.DEFINE_string('train_dir','pretrain.state','Pre-train state directory')
tf.app.flags.DEFINE_integer('max_steps', 500000, 'Max number of steps for pre-training')

def main(argv=None):
    if not _check_env():
        return
    
    print('Starting pre-training, storing training state in "%s"' % FLAGS.train_dir)

    model = Model()
    model.init_pre_train(
        batch_size = FLAGS.batch_size,
        state_dir = FLAGS.train_dir,
        max_steps = FLAGS.max_steps,
        threads = 16,
        examples_per_epoch = 28000,
        min_fraction = 0.4)
    
    model.pre_train()


def _check_env():
    if not tf.gfile.Exists(FLAGS.train_file):
        print('Requires pre-train file "%s"' % FLAGS.train_file)
        return False
    if tf.gfile.Exists(FLAGS.train_dir):
        print('Deleting %s...' % FLAGS.train_dir)
        tf.gfile.DeleteRecursively(FLAGS.train_dir)

    return True

if __name__ == '__main__':
    tf.app.run()

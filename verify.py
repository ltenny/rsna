import tensorflow as tf
import numpy as np
import sys
from utils.record import Record
import os
import uuid
from PIL import Image

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pre_train_file','pre_train.tfrecord','TFRecord containing pre-training features')
tf.app.flags.DEFINE_string('positives', 'positives_verify', 'Positives verify path')
tf.app.flags.DEFINE_string('negatives', 'negatives_verify', 'Negatives verify path')
tf.app.flags.DEFINE_boolean('grayscale',True,'If True, the TFRecord file is created with 1 channel, grayscale')

def _init_output_directories():
    if tf.gfile.Exists(FLAGS.positives):
        print('Deleting %s...' % FLAGS.positives)
        tf.gfile.DeleteRecursively(FLAGS.positives)
    tf.gfile.MakeDirs(FLAGS.positives)

    if tf.gfile.Exists(FLAGS.negatives):
        print('Deleting %s...' % FLAGS.negatives)
        tf.gfile.DeleteRecursively(FLAGS.negatives)
    tf.gfile.MakeDirs(FLAGS.negatives)



# main entry point
def main(argv=None):
    if FLAGS.grayscale:
        print('Grayscale not currently supported!')
        return
        
    _init_output_directories()
    batch_size = 100
    rec = Record(1024,1024,1 if FLAGS.grayscale else 3)
    with tf.Graph().as_default():
        image, label = rec.read_pre_train_record([FLAGS.pre_train_file])
        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=300, num_threads=10, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            image_batch, label_batch = sess.run([images, labels])

            for ndx in range(batch_size):
                img = image_batch[ndx]
                lbl = label_batch[ndx]
                print('Getting image for label %d' % lbl)
                img = img.astype(np.uint8)
                img = np.stack([img] * 3, axis = 2)
                print(img.shape)
                new_image = Image.fromarray(img)
                if lbl == 1:
                    path = FLAGS.positives
                else:
                    path = FLAGS.negatives
                new_image.save(os.path.join(path,'%s.jpg' % str(uuid.uuid4())))
            
            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    tf.app.run()
    
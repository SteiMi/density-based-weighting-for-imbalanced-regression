import os
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srcnn import srcnn

# model parameters
flags = tf.flags

# model hyperparamters
flags.DEFINE_string('hidden', '64,32,1', 'Number of units in hidden layer 1.')
flags.DEFINE_string('kernels', '9,3,5', 'Kernel size of layer 1.')
flags.DEFINE_integer('depth', 1, 'Number of input channels.')

# Model training parameters
flags.DEFINE_integer('num_epochs', 10000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('input_size', 31, 'Number of input channels.')

# when to save, plot, and test
flags.DEFINE_integer('save_step', 100, 'How often should I save the model')
flags.DEFINE_integer('test_step', 10, 'How often test steps are executed and printed')

# where to save things
flags.DEFINE_string('data_dir', 'data/train_tfrecords_2/', 'Data Location')
flags.DEFINE_string('test_dir', 'data/test/Set5_tfrecords_2', 'What should I be testing?')
flags.DEFINE_string('save_dir', 'results/', 'Where to save checkpoints.')
flags.DEFINE_string('log_dir', 'logs/', 'Where to save checkpoints.')

def _maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def read_and_decode(filename_queue, is_training=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.int64),
          'height':tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64)
      })

    with tf.device("/cpu:0"):
        if is_training:
            imgshape = [FLAGS.input_size, FLAGS.input_size, FLAGS.depth]
        else:
            depth = tf.cast(tf.reshape(features['depth'], []), tf.int32)
            width = tf.cast(tf.reshape(features['width'], []), tf.int32)
            height = tf.cast(tf.reshape(features['height'], []), tf.int32)
            imgshape = tf.stack([height, width, depth])

        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, imgshape)

        label = tf.decode_raw(features['label'], tf.float32)
        label = tf.reshape(label, imgshape)
        label_y = imgshape[0] - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
        label_x = imgshape[1] - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
        label = tf.slice(label, [FLAGS.padding, FLAGS.padding, 0], [label_y, label_x, -1])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(label, tf.float32) * (1. / 255) - 0.5
        return image, label

def inputs(train, batch_size, num_epochs=None):
    if train:
        files = [os.path.join(FLAGS.data_dir, f) for f in os.listdir(FLAGS.data_dir) if 'train' in f]
    else:
        files = [os.path.join(FLAGS.test_dir, f) for f in os.listdir(FLAGS.test_dir)]
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            files, num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue, is_training=train)

        # Shuffle the examples and collect them into batch_size batches.
        # We run this in two threads to avoid being a bottleneck.
        if train:
            image, label = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=8,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        else:
            image = tf.expand_dims(image, 0)
            label = tf.expand_dims(label, 0)
        return image,  label

def train():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        train_images, train_labels = inputs(True, FLAGS.batch_size, FLAGS.num_epochs)
        test_images, test_labels = inputs(False, FLAGS.batch_size, FLAGS.num_epochs)

        # set some placeholders
        x = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.depth),
                                                       name="input")
        y = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.depth),
                                                       name="label")
        is_training = tf.placeholder_with_default(True, (), name='is_training') 

        # build graph
        model = srcnn.SRCNN(x, y, FLAGS.HIDDEN_LAYERS, FLAGS.KERNELS,
                            is_training=is_training, input_depth=FLAGS.depth,
                            output_depth=FLAGS.depth,
                           learning_rate=1e-4, gpu=True)

        # initialize graph
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # how many images should we iterate through to test
        if 'set14' in FLAGS.test_dir.lower():
            test_iters = 14
        elif 'set5' in FLAGS.test_dir.lower():
            test_iters = 5
        else:
            test_iters = 1

        # for demo purposes this will keep the code simpler
        # in practice you'd want to feed train_images and train_labels in to SRCNN as a pipline
        def feed_dict(train=True):
            if train:
                im, lab = sess.run([train_images, train_labels])
            else:
                im, lab = sess.run([test_images, test_labels])
            return {x: im, y: lab, is_training: True}

        for step in range(FLAGS.num_epochs):
            _, train_loss = sess.run([model.opt, model.loss],
                    feed_dict=feed_dict(True))

            if step % FLAGS.test_step == 0:
                for j in range(test_iters):
                    #im, lab = sess.run([test_images, test_labels])
                    test_stats = sess.run([model.loss],
                        feed_dict=feed_dict(False))
                print "Step: %i, Train Loss: %2.4f, Test Loss: %2.4f" %\
                    (step, train_loss, test_stats[0])
            if step % FLAGS.save_step == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" % step))
        save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" %
                                                                                                                            step))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS._parse_flags()

    FLAGS.HIDDEN_LAYERS = [int(x) for x in FLAGS.hidden.split(",")]
    FLAGS.KERNELS = [int(x) for x in FLAGS.kernels.split(",")]
    FLAGS.label_size = FLAGS.input_size - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
    FLAGS.padding = abs(FLAGS.input_size - FLAGS.label_size) / 2

    file_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(file_dir, FLAGS.save_dir, "%s_%s_%i" % (
                        FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-"),
                        FLAGS.batch_size))
    FLAGS.log_dir = os.path.join(file_dir, FLAGS.log_dir)
    #FLAGS.data_dir = os.path.join(file_dir, FLAGS.data_dir)
    #FLAGS.test_dir = os.path.join(file_dir, FLAGS.test_dir)


    _maybe_make_dir(FLAGS.log_dir)
    _maybe_make_dir(os.path.dirname(SAVE_DIR))
    _maybe_make_dir(SAVE_DIR)
    train()

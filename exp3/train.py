import os
import time
import sys
import datetime as dt
import re

import numpy as np
import tensorflow as tf
import matplotlib as mpl
# Make sure matplotlib works when running in kubernetes cluster without X server
# See: https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
if os.uname()[0].lower() != 'darwin':
    mpl.use('Agg')
import matplotlib.pyplot as plt

# This imports configparser regardless of python2 or python3
try:
    import configparser as ConfigParser
except ImportError:
    import ConfigParser

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

base_srcnn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srcnn-tensorflow')
sys.path.append(base_srcnn)
from srcnn import srcnn
from tfreader import inputs_climate, get_filenames, read_and_decode
from utils import mkdir_p, init_mpl, set_size

flags = tf.flags
flags.DEFINE_string('config_file', 'config.ini', 'Configuration file with [SRCNN] section.')
flags.DEFINE_string('checkpoint_file', None, 'Any checkpoint with the same architecture as'\
                    'configured.')
flags.DEFINE_string('model_number', '1', 'Experiment-? in config file/')

# parse flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

## READ CONFIGURATION FILE
config = ConfigParser.ConfigParser()
config.read(FLAGS.config_file)

LAYER_SIZES = [int(k) for k in config.get('SRCNN', 'layer_sizes').split(",")]
KERNEL_SIZES = [int(k) for k in config.get('SRCNN', 'kernel_sizes').split(",")]
OUTPUT_DEPTH = LAYER_SIZES[-1]
AUX_DEPTH = int(config.get('SRCNN', 'aux_depth'))
LEARNING_RATE = float(config.get('SRCNN', 'learning_rate'))
TRAINING_ITERS = int(config.get('SRCNN', 'training_iters'))
BATCH_SIZE = int(config.get('SRCNN', 'batch_size'))
TRAINING_INPUT_SIZE = int(config.get('SRCNN', 'training_input_size'))
TRAINING_LR_INPUT_SIZE = int(config.get('SRCNN', 'training_lr_input_size'))
INPUT_DEPTH = int(config.get('SRCNN', 'training_input_depth'))
SAVE_STEP = int(config.get('SRCNN', 'save_step'))
TEST_STEP = int(config.get('SRCNN', 'test_step'))
KEEP_PROB = 1. - float(config.get('SRCNN', 'dropout_prob'))
WEIGHTED_LOSS = config.getboolean('SRCNN', 'weighted_loss')
ALPHA = float(config.get('SRCNN', 'alpha'))

# where to save and get data
DEEPSD_MODEL_NAME = config.get('DeepSD', 'model_name')
DATA_DIR = os.path.expanduser(config.get('Model-%s' % FLAGS.model_number, 'data_dir'))
MODEL_NAME = config.get('Model-%s' % FLAGS.model_number, 'model_name')
timestamp = str(int(time.time()))
curr_time = dt.datetime.now()

OUTPUT_DIR = os.path.expanduser(os.path.join(config.get('SRCNN', 'scratch'), DEEPSD_MODEL_NAME,  'outputs'))

SAVE_DIR = os.path.expanduser(os.path.join(config.get('SRCNN', 'scratch'), DEEPSD_MODEL_NAME, 'models',
                              "srcnn_%s_%s_%s" % (MODEL_NAME,
                              '-'.join([str(s) for s in LAYER_SIZES]),
                              '-'.join([str(s) for s in KERNEL_SIZES]))))


mkdir_p(SAVE_DIR)


def load_all_train_data():
    """
    This function returns all the densities and labels of the training data.

    I need this to calculate the mean weight beforehand, so that I can normalize my weights to a mean of 1.
    """
    filenames = get_filenames(DATA_DIR, is_training=True, with_densities=True)
    lr_size = int(TRAINING_INPUT_SIZE / 2)
    with tf.Session() as sess:
        # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        #                                         log_device_placement=False))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
        data = read_and_decode(filename_queue, True, INPUT_DEPTH, AUX_DEPTH, OUTPUT_DEPTH,
                               lr_shape=[lr_size, lr_size], hr_shape=[TRAINING_INPUT_SIZE, TRAINING_INPUT_SIZE],
                               densities=True, resize_img=False, seperate_img_and_aux=True)

        dens = data['dens']
        label = data['label']

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        densities = []
        labels = []
        try:
            while True:
                d, l = sess.run([dens, label])
                densities.append(d)
                labels.append(l)
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)

    # the new axis makes sure that I can reuse the crop function. The new axis emulates the batch_size dimension.
    np_densities = np.concatenate(densities, axis=2)[np.newaxis, ...]
    np_labels = np.concatenate(labels, axis=2)[np.newaxis, ...]

    np_densities = crop_train_tensor(np_densities)
    np_densities = np_densities.flatten()

    np_labels = crop_train_tensor(np_labels)
    np_labels = np_labels.flatten()
    return np_densities, np_labels


def calculate_w2(densities, epsilon=1e-6):
    w2 = np.maximum(epsilon, 1 - ALPHA * densities)
    return w2


def crop_train_tensor(tensor):
    # crop training labels
    border_size = int((sum(KERNEL_SIZES) - len(KERNEL_SIZES))/2)
    return tensor[:, border_size:-border_size, border_size:-border_size, :]


def train():

    with tf.Graph().as_default(), tf.device("/cpu:0"):
        # Get all densities to calculate mean weight for normalization of our weighted loss
        if WEIGHTED_LOSS:
            # Assume that there is latex if we are on MacOS
            use_latex_rendering = os.uname()[0].lower() == 'darwin'
            # Initialize matplotlib
            init_mpl(usetex=use_latex_rendering)

            print('Loading all densities...')
            densities, labels = load_all_train_data()

            idxs = labels.argsort()
            densities = densities[idxs]
            labels = labels[idxs]

            w2 = calculate_w2(densities)
            mean_weight = np.mean(w2)

            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)

            mpl.rcParams['agg.path.chunksize'] = 10000
            plt.figure(figsize=set_size(fraction=0.475))
            plt.xlabel('Precipitation (mm)')
            plt.ylabel('$p$')
            # Only plot every nth datapoint to speed things up and make the pdf readable
            nth = 100
            print(labels[::nth].shape)
            plt.hist(labels[::nth], bins=10, density=True, log=True)
            plt.plot(labels[::nth], densities[::nth], 'r-', label='$p\'(y)$')
            plt.plot(labels[::nth], (w2/mean_weight)[::nth], 'g-', label='$w^{%s}(y)$' % ALPHA)
            plt.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, 'weight_distribution.pdf'), format='pdf', bbox_inches='tight')

            print('Min/Mean/Max w2', np.min(w2), np.mean(w2), np.max(w2))
            print('Min/Mean/Max density', np.min(densities), np.mean(densities), np.max(densities))
            print('mean_weight', mean_weight)

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        errors = []

        # lets get data to iterate through
        # lr_size = int(TRAINING_INPUT_SIZE / 2)
        lr_size = TRAINING_LR_INPUT_SIZE
        train_data = inputs_climate(BATCH_SIZE, TRAINING_ITERS,
                        DATA_DIR, lr_shape=[lr_size, lr_size], lr_d=INPUT_DEPTH,
                        aux_d=AUX_DEPTH, is_training=True,
                        hr_shape=[TRAINING_INPUT_SIZE, TRAINING_INPUT_SIZE], hr_d=OUTPUT_DEPTH,
                        with_densities=WEIGHTED_LOSS)
                        # with_densities=True)

        if WEIGHTED_LOSS:
            train_images, train_labels, train_dens = train_data
        else:
            train_images, train_labels = train_data

        test_images, test_labels, test_times = inputs_climate(BATCH_SIZE, TRAINING_ITERS,
                        DATA_DIR, is_training=False, lr_d=INPUT_DEPTH, aux_d=1,
                        hr_d=OUTPUT_DEPTH)

        # crop training labels
        train_labels_cropped = crop_train_tensor(train_labels)
        if WEIGHTED_LOSS:
            train_dens_cropped = crop_train_tensor(train_dens)

        # set placeholders
        is_training = tf.placeholder_with_default(True, (), name='is_training')

        x = tf.cond(is_training, lambda: train_images, lambda: test_images)
        y = tf.cond(is_training, lambda: train_labels_cropped, lambda: test_labels)

        x = tf.identity(x, name='x')
        y = tf.identity(y, name='y')
        dens = tf.identity(train_dens_cropped, name='dens') if WEIGHTED_LOSS else None

        # Use SRCNN
        model = srcnn.SRCNN(x, y, LAYER_SIZES, KERNEL_SIZES, input_depth=INPUT_DEPTH,
                            learning_rate=LEARNING_RATE, upscale_factor=2,
                            is_training=is_training, gpu=True, densities=dens,
                            weighted_loss=WEIGHTED_LOSS, alpha=ALPHA, mean_weight=mean_weight)
        prediction = tf.identity(model.prediction, name='prediction')

        # initialize graph and start session
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False))
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # look for checkpoint
        if FLAGS.checkpoint_file is not None:
            try:
                checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_file)
                saver.restore(sess, checkpoint)
                print("Checkpoint", checkpoint)
            except tf.errors.InternalError as err:
                print("Warning: Could not find checkpoint", err)
                pass

        # start coordinator for data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # summary data
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SAVE_DIR + '/test', sess.graph)

        def feed_dict(train=True):
            return {is_training: train}

        #curr_step = int(sess.run(model.global_step))
        curr_step = 0
        for step in range(curr_step, TRAINING_ITERS+1):
            start_time = time.time()
            _, train_loss, train_rmse = sess.run([model.opt, model.loss, model.rmse],
                                                 feed_dict=feed_dict(True))
            duration = time.time() - start_time
            if step % TEST_STEP == 0:
                test_summary = sess.run(summary_op, feed_dict=feed_dict(True))
                train_writer.add_summary(test_summary, step)

                d = feed_dict(train=True)
                out = sess.run([model.loss, model.rmse, summary_op, model.x_norm], feed_dict=d)
                test_writer.add_summary(out[2], step)
                print("Step: %d, Examples/sec: %0.5f, Training Loss: %2.3f," \
                        " Train RMSE: %2.3f, Test RMSE: %2.4f" % \
                        (step, BATCH_SIZE/duration, train_loss, train_rmse, out[1]))

            if step % SAVE_STEP == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "srcnn.ckpt"))

        save_path = saver.save(sess, os.path.join(SAVE_DIR, "srcnn.ckpt"))

if __name__ == "__main__":
    train()

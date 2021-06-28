"""
This script adds densities to existing .tfrecords files created by prism.py.
We use this to apply DenseLoss.
"""
import sys
import os
import configparser as ConfigParser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tfreader import read_and_decode
from tfwriter import convert_to_tf
from target_relevance import TargetRelevance

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
OUTPUT_DEPTH = LAYER_SIZES[-1]
AUX_DEPTH = int(config.get('SRCNN', 'aux_depth'))
BATCH_SIZE = int(config.get('SRCNN', 'batch_size'))
TRAINING_INPUT_SIZE = int(config.get('SRCNN', 'training_input_size'))
INPUT_DEPTH = int(config.get('SRCNN', 'training_input_depth'))
# where to save and get data
data_dir = os.path.expanduser(config.get('Model-%s' % FLAGS.model_number, 'data_dir'))
lr_size = int(TRAINING_INPUT_SIZE / 2)

filenames = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if 'tfrecords' in f])

filenames = [f for f in filenames if 'train' in f and 'with_densities' not in f]

print(filenames)

# Go through all files to get the label for a kde
with tf.Session() as sess:
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    data = read_and_decode(filename_queue, True, INPUT_DEPTH, AUX_DEPTH, OUTPUT_DEPTH,
                           lr_shape=[lr_size, lr_size], hr_shape=[TRAINING_INPUT_SIZE, TRAINING_INPUT_SIZE],
                           densities=False, resize_img=False, seperate_img_and_aux=True)

    label = data['label']


    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    labels = []
    try:
        while True:
            l = sess.run([label])
            labels.append(l)
    except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

np_labels = np.concatenate(labels, axis=2).flatten().reshape(-1, 1)

# Flatten again
np_labels = np_labels.flatten()

# sort labels
np_labels = np.sort(np_labels, axis=0)
print(np_labels)

target_relevance = TargetRelevance(np_labels)

plt.hist(np_labels, bins=100, density=True)
plt.plot(target_relevance.x, target_relevance.y_dens, 'r-', label='densities')
plt.show()

print('-10', target_relevance.get_density(-10))
print('-0.5', target_relevance.get_density(-0.5))
print('0.0', target_relevance.get_density(0.0))
print('0.5', target_relevance.get_density(0.5))
print('10', target_relevance.get_density(10))
print('100', target_relevance.get_density(100))


# Load all files again but this time file by file not all at once so that I can still know which data point belongs to
# which file
for filename in filenames:
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        data = read_and_decode(filename_queue, True, INPUT_DEPTH, AUX_DEPTH, OUTPUT_DEPTH,
                               lr_shape=[lr_size, lr_size], hr_shape=[TRAINING_INPUT_SIZE, TRAINING_INPUT_SIZE],
                               densities=False, resize_img=False, seperate_img_and_aux=True)

        image = data['input']  # [:, :, 0]
        aux = data['aux']  # [:, :, 1]  # aux == elevation
        label = data['label']
        lat = data['lat']
        lon = data['lon']
        time = data['time']

        print(image.shape, aux.shape, label.shape, lat.shape, lon.shape, time.shape)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images = []
        auxs = []
        labels = []
        lats = []
        lons = []
        times = []
        densities = []
        try:
            while True:
                i, a, l, la, lo, t = sess.run([image, aux, label, lat, lon, time])
                # Map each value of the label matrix individually to densities
                d = np.vectorize(target_relevance.get_density)(l)
                images.append(i[np.newaxis, :, :]) 
                auxs.append(a[np.newaxis, :, :])
                labels.append(l[np.newaxis, :, :, :])
                lats.append(la[np.newaxis, :])
                lons.append(lo[np.newaxis, :])
                times.append(t)
                densities.append(d[np.newaxis, :, :, :])
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)

    order = list(range(len(images)))
    np.random.shuffle(order)
    np_images = np.concatenate(images, axis=0)[order]
    np_auxs = np.concatenate(auxs, axis=0)[order]
    np_labels = np.concatenate(labels, axis=0)[order]
    np_lats = np.concatenate(lats, axis=0)[order]
    np_lons = np.concatenate(lons, axis=0)[order]
    np_times = np.array(times)[order] 
    np_densities = np.concatenate(densities, axis=0)[order]

    print(np_images.shape, np_auxs.shape, np_labels.shape, np_lats.shape, np_lons.shape, np_times.shape, np_densities.shape)

    new_filename = filename.split('.')[0] + '_with_densities.tfrecords'
    print('Writing', new_filename)
    convert_to_tf(np_images, np_auxs, np_labels, np_lats, np_lons, np_times, new_filename, densities=np_densities)

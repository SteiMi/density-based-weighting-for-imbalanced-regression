import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

def read_and_decode(filename_queue, is_training,lr_d,aux_d,hr_d,
                    lr_shape=None, hr_shape=None, densities=False,
                    resize_img=True, seperate_img_and_aux=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feat_dict = {
            'hr_h': tf.FixedLenFeature([], tf.int64),
            'hr_w': tf.FixedLenFeature([], tf.int64),
            'lr_h': tf.FixedLenFeature([], tf.int64),
            'lr_w': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'img_in': tf.FixedLenFeature([], tf.string),
            'aux': tf.FixedLenFeature([], tf.string),
            'lat': tf.FixedLenFeature([], tf.string),
            'lon': tf.FixedLenFeature([], tf.string),
            'time': tf.FixedLenFeature([], tf.int64)
    }

    if densities:
        feat_dict['dens'] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(
        serialized_example,
        features=feat_dict)

    with tf.device("/cpu:0"):
        if is_training:
            hr_h, hr_w = hr_shape
            lr_h, lr_w = lr_shape
        else:
            hr_w = tf.cast(tf.reshape(features['hr_w'], []), tf.int32)
            hr_h = tf.cast(tf.reshape(features['hr_h'], []), tf.int32)

            lr_w = tf.cast(tf.reshape(features['lr_w'], []), tf.int32)
            lr_h = tf.cast(tf.reshape(features['lr_h'], []), tf.int32)

            #input_shape = tf.stack([lr_h, lr_w, lr_d])
            #aux_shape = tf.stack([hr_h, hr_w, aux_d])
            #label_shape = tf.stack([hr_h, hr_w, hr_d])

        img_in = tf.decode_raw(features['img_in'], tf.float32)
        img_in = tf.reshape(img_in, [lr_h, lr_w, lr_d])
        img_in = tf.cast(img_in, tf.float32)
        if resize_img:
            img_in = tf.image.resize_images(img_in, [hr_h, hr_w])
            img_in = tf.reshape(img_in, [hr_h, hr_w, lr_d])

        label = tf.decode_raw(features['label'], tf.float32)
        label = tf.reshape(label, [hr_h, hr_w, hr_d])
        label = tf.cast(label, tf.float32)

        aux = tf.decode_raw(features['aux'], tf.float32)
        aux = tf.reshape(aux, [hr_h, hr_w, aux_d])
        aux = tf.cast(aux, tf.float32)
        if not seperate_img_and_aux:
            img_in = tf.concat([img_in, aux], axis=2)

        lat = tf.decode_raw(features['lat'], tf.float32)
        lat = tf.reshape(lat, [hr_h])

        lon = tf.decode_raw(features['lon'], tf.float32)
        lon = tf.reshape(lon, [hr_w])

        out_dict = {"input": img_in, "label": label,
                "lat":lat, "lon":lon, "time": features['time']}

        if seperate_img_and_aux:
            out_dict['aux'] = aux

        if densities:
            dens = tf.decode_raw(features['dens'], tf.float32)
            dens = tf.reshape(dens, [hr_h, hr_w, hr_d])
            dens = tf.cast(dens, tf.float32)

            out_dict["dens"] = dens

        return out_dict


def get_filenames(data_dir, is_training=False, with_densities=False):
    filenames = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if 'tfrecords' in f])
    if is_training and with_densities:
        filenames = [f for f in filenames if 'train' in f and 'with_densities' in f]
    elif is_training and not with_densities:
        filenames = [f for f in filenames if 'train' in f and 'with_densities' not in f]
    else:
        filenames = [f for f in filenames if 'test' in f]

    return filenames


def inputs_climate(batch_size, num_epochs, data_dir, lr_d, aux_d, hr_d, lr_shape=None,
                   hr_shape=None, is_training=False, with_densities=False):
    filenames = get_filenames(data_dir, is_training=is_training, with_densities=with_densities)

    print(filenames)

    with tf.name_scope('input'), tf.device("/cpu:0"):
        filename_queue = tf.train.string_input_producer(filenames)
        data = read_and_decode(filename_queue, is_training, lr_d, aux_d, hr_d,
                               lr_shape=lr_shape, hr_shape=hr_shape, densities=with_densities)
        # what will happen to nan values? 
        if is_training and with_densities:
            images, labels, dens = tf.train.shuffle_batch([data['input'], data['label'], data['dens']],
                                           batch_size=batch_size,
                                           num_threads=8, capacity=2000 + 3*batch_size,
                                           min_after_dequeue=1000, allow_smaller_final_batch=True)
            return images, labels, dens
        elif is_training and not with_densities:
            images, labels = tf.train.shuffle_batch([data['input'], data['label']],
                                           batch_size=batch_size,
                                           num_threads=8, capacity=2000 + 3*batch_size,
                                           min_after_dequeue=1000, allow_smaller_final_batch=True)
            return images, labels
        else:
            images = tf.expand_dims(data['input'], 0)
            labels = tf.expand_dims(data['label'], 0)
            times = data['time']
            return images, labels, times

def fill_na(x, fillval=0):
    fill = tf.ones_like(x) * fillval
    return tf.select(tf.is_finite(x), x, fill)

def nanmean(x, axis=None):
    x_filled = fill_na(x, 0)
    x_sum = tf.reduce_sum(x_filled, reduction_indices=axis)
    x_count = tf.reduce_sum(tf.cast(tf.is_finite(x), tf.float32), reduction_indices=axis)
    return tf.div(x_sum, x_count)

def nanvar(x, axis=None):
    x_filled = fill_na(x, 0)
    x_count = tf.reduce_sum(tf.cast(tf.is_finite(x), tf.float32), reduction_indices=axis)
    x_mean = nanmean(x, axis=axis)
    x_ss = tf.reduce_sum((x_filled - x_mean)**2, reduction_indices=axis)
    return x_ss / x_count

def nan_batch_norm(inputs, decay=0.999, center=True, scale=False, epsilon=0.001,
        is_training=True, reuse=None, variables_collections=None, outputs_collections=None,
        trainable=False, scope=None):
    with variable_scope.variable_op_scope([inputs],
                    scope, 'NanBatchNorm', reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
          raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = list(range(inputs_rank - 1))
        params_shape = inputs_shape[-1:]
        beta, gamma = None, None
        if center:
          beta_collections = utils.get_variable_collections(variables_collections,
                                                            'beta')
          beta = variables.model_variable('beta',
                                          shape=params_shape,
                                          dtype=dtype,
                                          initializer=init_ops.zeros_initializer,
                                          collections=beta_collections,
                                          trainable=False)
        if scale:
          gamma_collections = utils.get_variable_collections(variables_collections,
                                                             'gamma')
          gamma = variables.model_variable('gamma',
                                           shape=params_shape,
                                           dtype=dtype,
                                           initializer=init_ops.ones_initializer,
                                           collections=gamma_collections,
                                           trainable=trainable)
        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean_collections = utils.get_variable_collections(
            variables_collections, 'moving_mean')
        moving_mean = variables.model_variable(
            'moving_mean',
            shape=params_shape,
            dtype=dtype,
            initializer=init_ops.zeros_initializer,
            trainable=False,
            collections=moving_mean_collections)
        moving_variance_collections = utils.get_variable_collections(
            variables_collections, 'moving_variance')
        moving_variance = variables.model_variable(
            'moving_variance',
            shape=params_shape,
            dtype=dtype,
            initializer=init_ops.ones_initializer,
            trainable=False,
            collections=moving_variance_collections)
        is_training_value = utils.constant_value(is_training)
        need_moments = is_training_value is None or is_training_value
        if need_moments:
            mean = nanmean(inputs, axis=axis)
            variance = nanvar(inputs, axis=axis)
            moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
        mean, variance = moving_mean, moving_variance
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs_shape)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

def inverse_batch_norm(inputs, mu, variance, beta, epsilon=0.001, name='predictions'):
    return tf.add((inputs-beta) * tf.sqrt(variance),  mu - epsilon, name=name)

def _prepend_edge(tensor, pad_amt, axis=1):
    '''
    This function is intented to add 'reflective' padding to a 4d Tensor across
        the height and width dimensions

    Parameters
    ----------
    tensor: Tensor with rank 4
    pad_amt: Integer
    axis: Integer
        Must be in (1,2)
    '''
    if axis not in (1, 2):
        raise ValueError("Axis must equal 0 or 1. Axis is set to %i" % axis)

    if axis == 1:
        concat_dim = 2
    else:
        concat_dim = 1

    begin = [0, 0, 0, 0]
    end = [-1, -1, -1, -1]
    end[axis] = 1

    edges = pad_amt*[tf.slice(tensor,begin,end)]
    if len(edges) > 1:
        padding = tf.concat(axis, edges)
    else:
        padding = edges[0]

    tensor_padded = tf.concat(axis, [padding, tensor])
    return tensor_padded

def _append_edge(tensor, pad_amt, axis=1):
    '''
    This function is intented to add 'reflective' padding to a 4d Tensor across
        the height and width dimensions

    Parameters
    ----------
    tensor: Tensor with rank 4
    pad_amt: Integer
    axis: Integer
        Must be in (1,2)
    '''
    if axis not in (1, 2):
        raise ValueError("Axis must equal 0 or 1. Axis is set to %i" % axis)

    if axis == 1:
        concat_dim = 2
    else:
        concat_dim = 1

    begin = [0, 0, 0, 0]
    end = [-1, -1, -1, -1]
    begin[axis] = tf.shape(tensor)[axis]-1 # go to the end

    edges = pad_amt*[tf.slice(tensor,begin,end)]

    if len(edges) > 1:
        padding = tf.concat(axis, edges)
    else:
        padding = edges[0]

    tensor_padded = tf.concat(axis, [tensor, padding])
    return tensor_padded

def replicate_padding(tensor, pad_amt):
    if isinstance(pad_amt, int):
        pad_amt = [pad_amt] * 2
    for axis, p in enumerate(pad_amt):
        tensor = _prepend_edge(tensor, p, axis=axis+1)
        tensor = _append_edge(tensor, p, axis=axis+1)
    return tensor

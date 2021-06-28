from __future__ import print_function
import os
import time
import sys

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import utils


def _maybe_pad_x(x, padding, is_training):
    if padding == 0:
        x_pad = x
    elif padding > 0:
        x_pad = tf.cond(is_training, lambda: x,
                        lambda: utils.replicate_padding(x, padding))
    else:
        raise ValueError("Padding value %i should be greater than or equal to 1" % padding)
    return x_pad


class SRCNN:
    def __init__(self, x, y, layer_sizes, filter_sizes, input_depth=1,
                 learning_rate=1e-4,
                 gpu=True, upscale_factor=2, output_depth=1, is_training=True,
                 densities=None, weighted_loss=False, alpha=1.0, mean_weight=None):
        '''
        Args:
            layer_sizes: Sizes of each layer
            filter_sizes: List of sizes of convolutional filters
            input_depth: Number of channels in input
        '''
        self.x = x
        self.y = y
        self.is_training = is_training
        self.upscale_factor = upscale_factor
        self.layer_sizes = layer_sizes
        self.filter_sizes = filter_sizes
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.learning_rate = learning_rate
        self.dens = densities
        self.weighted_loss = weighted_loss
        self.alpha = alpha
        self.mean_weight = mean_weight

        if self.weighted_loss:
            assert self.mean_weight is not None, "Invalid mean_weight: %r" % self.mean_weight
            assert self.mean_weight >= 0, "Invalid mean_weight: %r" % self.mean_weight
            assert self.alpha is not None, "Invalid alpha: %r" % self.alpha
            assert self.alpha >= 0, "Invalid alpha: %r" % self.alpha

        if gpu:
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                   100000, 0.96)
        self._build_graph()

    def _normalize(self):
        with tf.variable_scope("normalize_inputs") as scope:
            self.x_norm = tf.contrib.layers.batch_norm(self.x, trainable=False,
                                epsilon=1e-6,
                               updates_collections=None, center=False,
                               scale=False, is_training=self.is_training)
        with tf.variable_scope("normalize_labels") as scope:
            self.y_norm = tf.contrib.layers.batch_norm(self.y, trainable=False,
                                    epsilon=1e-6, updates_collections=None,
                                  scale=False, is_training=self.is_training)
            scope.reuse_variables()
            self.y_mean = tf.get_variable('BatchNorm/moving_mean')
            self.y_variance = tf.get_variable('BatchNorm/moving_variance')
            self.y_beta = tf.get_variable('BatchNorm/beta')

    def _inference(self, X):
        for i, k in enumerate(self.filter_sizes):
            with tf.variable_scope("hidden_%i" % i) as scope:
                if i == (len(self.filter_sizes)-1):
                    activation = None
                else:
                    activation = tf.nn.relu
                pad_amt = (k-1)/2
                X = _maybe_pad_x(X, pad_amt, self.is_training)
                X = tf.layers.conv2d(X, self.layer_sizes[i], k, activation=activation)
        return X

    def _loss(self, predictions):
        with tf.name_scope("loss"):
            err = tf.square(predictions - self.y_norm)
            err_filled = utils.fill_na(err, 0)

            raw_relevance = 1 - self.alpha_var * self.dens
            eps = 1e-6
            self.relevance = tf.clip_by_value(raw_relevance, eps, float("inf")) / self.mean_weight

            if self.weighted_loss:
                print('weighted loss activated')
                err_filled = self.relevance * err_filled

            finite_count = tf.reduce_sum(tf.cast(tf.is_finite(err), tf.float32))
            mse = tf.reduce_sum(err_filled) / finite_count
            return mse

    def _optimize(self):
        opt1 = tf.train.AdamOptimizer(self.learning_rate)
        opt2 = tf.train.AdamOptimizer(self.learning_rate*0.1)

        # compute gradients irrespective of optimizer
        grads = opt1.compute_gradients(self.loss)

        # apply gradients to first n-1 layers 
        opt1_grads = [v for v in grads if "hidden_%i" % (len(self.filter_sizes)-1)
                    not in v[0].op.name]
        opt2_grads = [v for v in grads if "hidden_%i" % (len(self.filter_sizes)-1)
                    in v[0].op.name]

        self.opt = tf.group(opt1.apply_gradients(opt1_grads, global_step=self.global_step),
                            opt2.apply_gradients(opt2_grads))

    def _summaries(self):
        tf.contrib.layers.summarize_tensors(tf.trainable_variables())
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('rmse', self.rmse)

    def _build_graph(self):
        self._normalize()

        self.alpha_var = tf.constant(self.alpha, dtype=tf.float32, name='alpha')
        self.mean_weight = tf.constant(self.mean_weight, dtype=tf.float32, name='mean_weight')

        with tf.device(self.device):
            _prediction_norm = self._inference(self.x_norm)
            self.loss = self._loss(_prediction_norm)

            self._optimize()
        self.prediction = _prediction_norm * tf.sqrt(self.y_variance) - self.y_mean
        self.rmse = tf.sqrt(utils.nanmean(tf.square(self.prediction - self.y)),
                            name='rmse')
        self._summaries()

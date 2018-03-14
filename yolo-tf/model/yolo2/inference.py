"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import inspect
import tensorflow as tf
import tensorflow.contrib.slim as slim
from ..yolo.function import leaky_relu
from .function import reorg


def tiny(net, classes, num_anchors, training=False, center=True):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net, slim.variable('biases', shape=[tf.shape(net)[-1]], initializer=tf.zeros_initializer()))
        return net
    scope = __name__.split('.')[-2] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), normalizer_fn=batch_norm, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        channels = 16
        for _ in range(5):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, stride=1, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None, scope='%s/conv' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net

TINY_DOWNSAMPLING = (2 ** 5, 2 ** 5)


def _tiny(net, classes, num_anchors, training=False):
    return tiny(net, classes, num_anchors, training, False)

_TINY_DOWNSAMPLING = (2 ** 5, 2 ** 5)


def darknet(net, classes, num_anchors, training=False, center=True):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net, slim.variable('biases', shape=[tf.shape(net)[-1]], initializer=tf.zeros_initializer()))
        return net
    scope = __name__.split('.')[-2] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], normalizer_fn=batch_norm, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        channels = 32
        for _ in range(2):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        for _ in range(2):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        passthrough = tf.identity(net, name=scope + '/passthrough')
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        # downsampling finished
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1

        # layer 24:
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        passthrough2 = tf.identity(net, name=scope + '/passthrough2')
        index += 1

        # channels = 1024 at this point

        # layer 25: route layer 16: feed forward the passthrough layer:
        # reconfigure inputs weights of passthrough layer (layer 16) from 26x26x256 to 13x13x1024
        with tf.name_scope(scope):
            passthrough.inputs = net.get_shape()
            _net = reorg(passthrough,stride=1,name='%s/route%d' % (scope, index))
        index += 1

        # layer 26: 1x1 convl of 26x26x512 -> 26x26x64to 64 channels: 
        net = slim.layers.conv2d(_net, 64, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1

        # layer 27: reorg layer 26 by stride 2: from 26x26x64 to 13x13x256
        with tf.name_scope(scope):
            _net = reorg(net, stride=2, name='%s/reorg%d' % (scope, index))
        index += 1

        # layer 28: route layers 27 and 24: 
        net = tf.concat([_net, passthrough2], 3, name='%s/concat%d' % (scope, index))
        index += 1

        # layer 29: 3x3 convolve 13x13x1280 to 13x13x1024
        net = slim.layers.conv2d(net, 1024, scope='%s/conv%d' % (scope, index))
        # index += 1

    # layer 30: convolve with channels = #Anchors * (5+#classes)
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None, scope='%s/conv' % scope)

    # layer 31: copy output toresult:
    net = tf.identity(net, name='%s/output' % scope)

    return scope, net

DARKNET_DOWNSAMPLING = (2 ** 5, 2 ** 5)


def _darknet(net, classes, num_anchors, training=False):
    return darknet(net, classes, num_anchors, training, False)

_DARKNET_DOWNSAMPLING = (2 ** 5, 2 ** 5)

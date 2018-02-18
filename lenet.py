import functools
import tensorflow as tf

import tensorflowmodelzoo as zoo
from tensorflow_model import TensorFlowModel


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    Learning TensorFlow, pp 212.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class LeNetTensorFlowModel(TensorFlowModel):

    def __init__(self):

        # Build placeholders values which change during execution.
        self.stimulus_placeholder = tf.placeholder(tf.float32)
        self.target_placeholder = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)

        self.inference

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def inference(self, input=None):
        '''
        input: tensor of input image. if none, uses instantiation input
        output: tensor of computed logits
        '''

        ###############################

        zoo.print_tensor_shape(self.stimulus_placeholder, 'images shape')
        zoo.print_tensor_shape(self.target_placeholder, 'label shape')

        # Convolution layer.
        with tf.name_scope('Conv1'):

            # weight variable 4d tensor, first two dims are patch (kernel) size
            # 3rd dim is number of input channels, 4th dim is output channels
            W_conv1 = self._weight_variable([5, 5, 1, 32])
            b_conv1 = self._bias_variable([32])
            h_conv1 = tf.nn.relu(self._conv2d(self.stimulus_placeholder, W_conv1) + b_conv1)
            zoo.print_tensor_shape(h_conv1, 'Conv1 shape')

        # Pooling layer.
        with tf.name_scope('Pool1'):

            h_pool1 = self._max_pool_2x2(h_conv1)
            zoo.print_tensor_shape(h_pool1, 'MaxPool1 shape')

        # Conv layer.
        with tf.name_scope('Conv2'):

            W_conv2 = self._weight_variable([5, 5, 32, 64])
            b_conv2 = self._bias_variable([64])
            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            zoo.print_tensor_shape(h_conv2, 'Conv2 shape')

        # Pooling layer.
        with tf.name_scope('Pool2'):

            h_pool2 = self._max_pool_2x2(h_conv2)
            zoo.print_tensor_shape(h_pool2, 'MaxPool2 shape')

        # Fully-connected layer.
        with tf.name_scope('fully_connected1'):

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            zoo.print_tensor_shape(h_pool2_flat, 'MaxPool2_flat shape')

            W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self._bias_variable([1024])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            zoo.print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # Dropout layer.
        with tf.name_scope('dropout'):

            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            W_fc2 = self._weight_variable([1024, 10])
            b_fc2 = self._bias_variable([10])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            zoo.print_tensor_shape(readout, 'readout shape')

        return readout
        ###############################

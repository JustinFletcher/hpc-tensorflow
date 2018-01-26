import tensorflow as tf


class TensorFlowModel(object):

    def inference(self):

        raise NotImplementedError

    def _add_variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor
        (for TensorBoard visualization)."""

        with tf.name_scope('summaries'):

            mean = tf.reduce_mean(var)

            tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):

                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar('stddev', stddev)

            tf.summary.scalar('max', tf.reduce_max(var))

            tf.summary.scalar('min', tf.reduce_min(var))

            tf.summary.histogram('histogram', var)

        return()

    def _weight_variable(self, shape):

        initial = tf.truncated_normal(shape, stddev=0.1)
        self._add_variable_summaries(initial)
        return tf.Variable(initial)

    def _bias_variable(self, shape):

        initial = tf.constant(0.1, shape=shape)
        self._add_variable_summaries(initial)
        return tf.Variable(initial)

    def _conv2d(self, x, W):

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

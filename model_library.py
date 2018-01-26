
import os
import sys
import time
import argparse
import functools
import numpy as np
import tensorflow as tf


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


# function to print the tensor shape.  useful for debugging
def print_tensor_shape(tensor, string):
    '''
    input: tensor and string to describe it
    '''

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())


class TensorFlowBatchProducer(object):

    def get_batch_ops(self):

        raise NotImplementedError


class MNISTTensorFlowBatchProducer(TensorFlowBatchProducer):

    def __init__(self,
                 data_dir,
                 train_file,
                 val_file,
                 input_size,
                 label_size):

        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.label_size = label_size
        self.input_size = input_size

    def _read_and_decode_mnist(self, filename_queue):

        # Instantiate a TFRecord reader.
        reader = tf.TFRecordReader()

        # Read a single example from the input queue.
        _, serialized_example = reader.read(filename_queue)

        # Parse that example into features.
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.input_size])

        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label_batch = features['label']

        label = tf.one_hot(label_batch,
                           self.label_size,
                           on_value=1.0,
                           off_value=0.0)

        return image, label

    def get_train_batch_ops(self,
                            batch_size=128,
                            capacity=10100,
                            num_threads=16,
                            min_after_dequeue=100):

        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.train_file)

        # Create an input scope for the graph.
        with tf.name_scope('train_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename],
                                                            capacity=1)

            # Even when reading in multiple threads, share the filename queue.
            image, label = self._read_and_decode_mnist(filename_queue)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                capacity=capacity,
                num_threads=num_threads,
                min_after_dequeue=min_after_dequeue)

        return images, sparse_labels

    def get_val_batch_ops(self,
                          batch_size=10000,
                          capacity=10100.0,
                          num_threads=16,
                          min_after_dequeue=100):

        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.val_file)

        # Create an input scope for the graph.
        with tf.name_scope('val_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename],
                                                            capacity=1)

            # Even when reading in multiple threads, share the filename queue.
            image, label = self._read_and_decode_mnist(filename_queue)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                capacity=capacity,
                num_threads=num_threads,
                min_after_dequeue=min_after_dequeue)

        return images, sparse_labels


class TensorFlowModelZoo(object):

    def get_model(self, model_name):

        if model_name == 'lenet':

            tfmodel = LeNetTensorFlowModel()

            return(tfmodel)

        else:

            print(model_name + " is not a recognized model name.")
            raise NotImplementedError


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

        print_tensor_shape(self.stimulus_placeholder, 'images shape')
        print_tensor_shape(self.target_placeholder, 'label shape')

        # resize the image tensors to add channels, 1 in this case
        # required to pass the images to various layers upcoming in the graph
        images_re = tf.reshape(self.stimulus_placeholder, [-1, 28, 28, 1])
        print_tensor_shape(images_re, 'reshaped images shape')

        # Convolution layer.
        with tf.name_scope('Conv1'):

            # weight variable 4d tensor, first two dims are patch (kernel) size
            # 3rd dim is number of input channels, 4th dim is output channels
            W_conv1 = self._weight_variable([5, 5, 1, 32])
            b_conv1 = self._bias_variable([32])
            h_conv1 = tf.nn.relu(self._conv2d(images_re, W_conv1) + b_conv1)
            print_tensor_shape(h_conv1, 'Conv1 shape')

        # Pooling layer.
        with tf.name_scope('Pool1'):

            h_pool1 = self._max_pool_2x2(h_conv1)
            print_tensor_shape(h_pool1, 'MaxPool1 shape')

        # Conv layer.
        with tf.name_scope('Conv2'):

            W_conv2 = self._weight_variable([5, 5, 32, 64])
            b_conv2 = self._bias_variable([64])
            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            print_tensor_shape(h_conv2, 'Conv2 shape')

        # Pooling layer.
        with tf.name_scope('Pool2'):

            h_pool2 = self._max_pool_2x2(h_conv2)
            print_tensor_shape(h_pool2, 'MaxPool2 shape')

        # Fully-connected layer.
        with tf.name_scope('fully_connected1'):

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            print_tensor_shape(h_pool2_flat, 'MaxPool2_flat shape')

            W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self._bias_variable([1024])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # Dropout layer.
        with tf.name_scope('dropout'):

            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            W_fc2 = self._weight_variable([1024, 10])
            b_fc2 = self._bias_variable([10])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            print_tensor_shape(readout, 'readout shape')

        return readout
        ###############################


class ModelTrainer(object):

    def __init__(self,
                 model,
                 data,
                 learning_rate):

        # Internalize instantiation parameters
        self.model = model
        self.data = data
        self.learning_rate = learning_rate

        # Register instance methods, building the computational graph.
        self.loss
        self.optimize
        self.error

    @define_scope
    def loss(self):

        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.model.target_placeholder,
            logits=self.model.inference,
            name='xentropy')

        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        return(loss)

    @define_scope
    def optimize(self):

        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.model.target_placeholder,
            logits=self.model.inference,
            name='xentropy')

        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        # Minimize the loss by incrementally changing trainable variables.
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    @define_scope
    def error(self):

        mistakes = tf.not_equal(tf.argmax(self.model.target_placeholder, 1),
                                tf.argmax(self.model.inference, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        # tf.summary.scalar('error', error)
        return(error)


def example_usage(_):

    # Clear the log directory, if it exists.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Reset the default graph.
    tf.reset_default_graph()

    # Declare experimental measurement variables as lists.
    steps = []
    val_losses = []
    val_errors = []
    train_losses = []
    train_errors = []
    mean_running_times = []

    print("------------model_output-------------")

    zoo = TensorFlowModelZoo()

    model = zoo.get_model('lenet')

    # TensorFlowModelZoo.get_model() ?

    batch_producer = MNISTTensorFlowBatchProducer(FLAGS.data_dir,
                                                  FLAGS.train_file,
                                                  FLAGS.validation_file,
                                                  FLAGS.input_size,
                                                  FLAGS.label_size)

    model_trainer = ModelTrainer(model=model,
                                 data=batch_producer,
                                 learning_rate=FLAGS.learning_rate)

    print("-------------------------------------")

    # Get input data.
    # TODO: Move this into the class, and use tf.data.Dataset API interfaces.
    (image_batch, label_batch) = batch_producer.get_train_batch_ops(
        batch_size=FLAGS.train_batch_size,
        capacity=10100.0,
        num_threads=FLAGS.train_enqueue_threads,
        min_after_dequeue=100)

    (val_image_batch, val_label_batch) = batch_producer.get_val_batch_ops(
        batch_size=FLAGS.val_batch_size,
        capacity=10100.0,
        num_threads=FLAGS.val_enqueue_threads,
        min_after_dequeue=100)

    # Merge the summary.
    tf.summary.merge_all()

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=600.0)

    with sv.managed_session() as sess:

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir +
        #                                      '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        # Declare timekeeping vars.
        running_times = [0.0001]
        running_time = 0

        print("------------training_output-------------")

        # Print a line for debug.
        print('step | train_loss | train_error | val_loss |' +
              ' val_error | t | total_time')

        # Load the validation set batch into memory.
        val_images, val_labels = sess.run([val_image_batch, val_label_batch])

        # Make a dict to load the val batch onto the placeholders.
        val_dict = {model.stimulus_placeholder: val_images,
                    model.target_placeholder: val_labels,
                    model.keep_prob: 1.0}

        time.sleep(FLAGS.pause_time)

        # Iterate until max steps.
        for i in range(FLAGS.max_steps):

            # Check for break.
            if sv.should_stop():
                break

            start_time = time.time()

            # If it is a batch refresh interval, refresh the batch.
            if((i % FLAGS.batch_interval == 0) or (i == 0)):

                # Update the batch.
                train_images, train_labels = sess.run([image_batch,
                                                       label_batch])

            # Make a dict to load the batch onto the placeholders.
            train_dict = {model.stimulus_placeholder: train_images,
                          model.target_placeholder: train_labels,
                          model.keep_prob: FLAGS.keep_prob}

            # If we have reached a testing interval, test.
            if (i % FLAGS.test_interval == 0):

                # Compute error over the training set.
                train_error = sess.run(model_trainer.error, feed_dict=train_dict)

                # Compute loss over the training set.
                train_loss = sess.run(model_trainer.loss, feed_dict=train_dict)

                # Compute error over the validation set.
                val_error = sess.run(model_trainer.error, feed_dict=val_dict)

                # Compute loss over the validation set.
                val_loss = sess.run(model_trainer.loss, feed_dict=val_dict)

                # Store the data we wish to manually report.
                steps.append(i)
                train_losses.append(train_loss)
                train_errors.append(train_error)
                val_losses.append(val_loss)
                val_errors.append(val_error)

                mean_running_time = np.mean(running_times)
                mean_running_times.append(mean_running_time)

                # Print relevant values.
                print('%d | %.6f | %.2f | %.6f | %.2f | %.6f | %.2f'
                      % (i,
                         train_loss,
                         train_error,
                         val_loss,
                         val_error,
                         np.mean(running_times),
                         np.sum(running_times)))

                # Reset running times measurment
                running_times = []

            # Optimize the model.
            sess.run(model_trainer.optimize, feed_dict=train_dict)

            # train_writer.add_summary(summary, i)

            # Update timekeeping variables.
            running_time = time.time() - start_time
            running_times.append(running_time)

        print("----------------------------------------")

        # Close the summary writers.
        # test_writer.close()
        # train_writer.close()
        sv.stop()
        sess.close()

    return()


if __name__ == '__main__':


    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--log_dir', type=str,
                        default='../log/tensorflow_experiment/templog',
                        help='Summaries log directory.')

    parser.add_argument('--pause_time', type=float,
                        default=0.0,
                        help='Number of seconds to pause before execution.')

    parser.add_argument('--log_filename', type=str,
                        default='deep_sa_generalization_experiment.csv',
                        help='Summaries log directory.')

    parser.add_argument('--keep_prob', type=float,
                        default=1.0,
                        help='Keep probability for output layer dropout.')

    parser.add_argument('--train_batch_size', type=int,
                        default=128,
                        help='Training set batch size.')

    parser.add_argument('--batch_interval', type=int,
                        default=1,
                        help='Interval between training batch refresh.')

    parser.add_argument('--max_steps', type=int, default=100,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=10,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    # These flags specify the data used in the experiment.
    parser.add_argument('--data_dir', type=str,
                        default='../data/mnist',
                        help='Directory from which to pull data TFRecords.')

    parser.add_argument('--train_file', type=str,
                        default='train.tfrecords',
                        help='Training dataset filename.')

    parser.add_argument('--validation_file', type=str,
                        default='validation.tfrecords',
                        help='Validation dataset filename.')

    parser.add_argument('--input_size', type=int,
                        default=28 * 28,
                        help='Dimensionality of the input space.')

    parser.add_argument('--label_size', type=int,
                        default=10,
                        help='Dimensinoality of the output space.')

    parser.add_argument('--val_batch_size', type=int,
                        default=10000,
                        help='Validation set batch size.')

    # These flags control the input pipeline threading.
    parser.add_argument('--val_enqueue_threads', type=int,
                        default=32,
                        help='Number of threads to enqueue val examples.')

    parser.add_argument('--train_enqueue_threads', type=int,
                        default=128,
                        help='Number of threads to enqueue train examples.')

    # These flags specify placekeeping variables.
    parser.add_argument('--rep_num', type=int,
                        default=0,
                        help='Flag identifying the repitition number.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=example_usage, argv=[sys.argv[0]] + unparsed)

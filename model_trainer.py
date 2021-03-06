
import sys
import time
import argparse
import functools
import numpy as np
import tensorflow as tf

import tensorflowmodelzoo as zoo
from mnist_batch_producer import MNISTTensorFlowBatchProducer


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

    model_zoo = zoo.TensorFlowModelZoo()
    # data_zoo = zoo.TensorFlowDataZoo()

    model = model_zoo.get_model('lenet')

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

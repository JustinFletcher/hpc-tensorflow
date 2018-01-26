
import os
import tensorflow as tf

from tensorflow_batch_producer import TensorFlowBatchProducer


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

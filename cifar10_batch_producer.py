
import os
import tensorflow as tf

from tensorflow_batch_producer import TensorFlowBatchProducer

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


###############################################################################
# Data processing
###############################################################################

# def parse_record(raw_record, is_training):
#     """Parse CIFAR-10 image and label from a raw record."""
#     # Convert bytes to a vector of uint8 that is record_bytes long.
#     record_vector = tf.decode_raw(raw_record, tf.uint8)

#     # The first byte represents the label, which we convert from uint8 to int32
#     # and then to one-hot.
#     label = tf.cast(record_vector[0], tf.int32)
#     label = tf.one_hot(label, _NUM_CLASSES)

#     # The remaining bytes after the label represent the image, which we reshape
#     # from [depth * height * width] to [depth, height, width].
#     depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
#                              [_NUM_CHANNELS, _HEIGHT, _WIDTH])

#     # Convert from [depth, height, width] to [height, width, depth],
#     # and cast as float32.
#     image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

#     image = preprocess_image(image, is_training)

#     return image, label


# def preprocess_image(image, is_training):
#     """Preprocess a single image of layout [height, width, depth]."""
#     if is_training:
#         # Resize the image to add four extra pixels on each side.
#         image = tf.image.resize_image_with_crop_or_pad(image,
#                                                        _HEIGHT + 8,
#                                                        _WIDTH + 8)

#         # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
#         image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

#         # Randomly flip the image horizontally.
#         image = tf.image.random_flip_left_right(image)

#     # Subtract off the mean and divide by the variance of the pixels.
#     image = tf.image.per_image_standardization(image)
#     return image


class CIFAR10TensorFlowBatchProducer(TensorFlowBatchProducer):

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

    def _preprocess_image(self, image, is_training):
        """Preprocess a single image of layout [height, width, depth]."""
        if is_training:
            # Resize the image to add four extra pixels on each side.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           _HEIGHT + 8,
                                                           _WIDTH + 8)

            # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
            image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)
        return image

    def _parse_record(self, raw_record, is_training):
        """Parse CIFAR-10 image and label from a raw record."""
        # Convert bytes to a vector of uint8 that is record_bytes long.
        record_vector = tf.decode_raw(raw_record, tf.uint8)

        # The first byte represents the label, which we convert from uint8 to int32
        # and then to one-hot.
        label = tf.cast(record_vector[0], tf.int32)
        label = tf.one_hot(label, _NUM_CLASSES)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                                 [_NUM_CHANNELS, _HEIGHT, _WIDTH])

        # Convert from [depth, height, width] to [height, width, depth],
        # and cast as float32.
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        image = self._preprocess_image(image, is_training)

        return image, label

    def _read_and_decode_cifar10(self, filename_queue, is_training):

        # # Instantiate a TFRecord reader.
        # reader = tf.TFRecordReader()

        # # Read a single example from the input queue.
        # _, serialized_example = reader.read(filename_queue)

        # # Parse that example into features.
        # raw_record = tf.parse_single_example(
        #     serialized_example,
        #     # Defaults are not specified since both keys are required.
        #     features={
        #         'image': tf.FixedLenFeature([], tf.string),
        #         'label': tf.FixedLenFeature([], tf.int64),
        #     })

        # # image, label = self._parse_record(raw_record, is_training)
        # image, label = self._parse_record(raw_record, is_training)

        # Instantiate a TFRecord reader.
        reader = tf.TFRecordReader()

        # Read a single example from the input queue.
        _, serialized_example = reader.read(filename_queue)

        # Parse that example into features.
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([self.input_size])

        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        # resize the image tensors to add channels, 1 in this case
        # required to pass the images to various layers upcoming in the graph
        image = tf.reshape(image, [32, 32, 3])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label_batch = features['label']

        label = tf.one_hot(label_batch,
                           self.label_size,
                           on_value=1.0,
                           off_value=0.0)

        return image, label

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
            image, label = self._read_and_decode_cifar10(filename_queue,
                                                         is_training=True)

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
            image, label = self._read_and_decode_cifar10(filename_queue,
                                                         is_training=False)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in many threads to avoid being a bottleneck.
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                capacity=capacity,
                num_threads=num_threads,
                min_after_dequeue=min_after_dequeue)

        return images, sparse_labels

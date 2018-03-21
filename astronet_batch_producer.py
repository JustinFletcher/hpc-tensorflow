
import argparse
import os
import sys
import numpy as np

import tensorflow as tf
import utils

from tensorflow_batch_producer import TensorFlowBatchProducer
from tfexample_decoder import BoundingBox
from utils.visualization_utils import save_image_array_as_png 
from utils.visualization_utils import visualize_boxes_and_labels_on_image_array 

FLAGS = None

# HEIGHT =512
# WIDTH = 512
# DEPTH = 3
# ONE = 1

# NUM_CLASSES = 1 # 'satellite'

class AstroNetBatchProducer(TensorFlowBatchProducer):

    def __init__(self,
                 data_dir,
                 train_file,
                 val_file,
                 test_file):

        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.keys_to_features = {
                'image/height': tf.FixedLenFeature([], tf.int64, 1),
                'image/width': tf.FixedLenFeature([], tf.int64, 1),
                'image/depth': tf.FixedLenFeature([], tf.int64, 1),
                'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                'image/object/class/text': tf.VarLenFeature(tf.string),
                'image/object/class/label': tf.VarLenFeature(tf.int64)
            }


    def _read_and_decode_astronet(self, filename_queue):

        # Instantiate a TFRecord reader.
        reader = tf.TFRecordReader()

        # Read a single example from the input queue.
        _, serialized_example = reader.read(filename_queue)

        # Parse the example into features.
        features = tf.parse_single_example(
            serialized_example,
            self.keys_to_features)
            # })

        # Convert from a scalar string tensor to a uint8 tensor with shape [height,width,channels].
        image = tf.decode_raw(features['image/encoded'], tf.uint8)

        height = tf.cast(features['image/height'], tf.int32)
        width = tf.cast(features['image/width'], tf.int32)
        depth = tf.cast(features['image/depth'], tf.int32)
    
        # print('height',height)
        # print('width',width)
        # print('depth',depth)
        image_shape = tf.stack([height, width, depth])       
        image = tf.reshape(image, image_shape)
        # image = tf.reshape(image, [512,512,3])
        # print('image.shape',image.shape)

        label = tf.cast(features['image/object/class/label'], tf.int32)
        class_name = tf.cast(features['image/object/class/text'], tf.string)
        file_name = tf.cast(features['image/filename'], tf.string)

        bbox = BoundingBox(keys=None, prefix='image/object/bbox/').tensors_to_item(features)

        return image, height, width, depth, label, class_name, file_name, bbox

    def get_train_batch_ops(self,
                            batch_size=128,
                            capacity=11430,
                            num_threads=16,
                            min_after_dequeue=100):

        print('In get_train_batch_ops:')
        
        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.train_file)

        # Create an input scope for the graph.
        with tf.name_scope('train_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename], capacity=1)

            # image_single,\
            # height_single,\
            # width_single, \
            # depth_single,\
            # label_single,\
            # class_name_single,\
            # file_name_single,\
            # bbox_single = self._read_and_decode_astronet(filename_queue)

            # set the shapes of the tensors:
            # image_single.set_shape([HEIGHT,WIDTH,DEPTH])
            # height_single.set_shape([ONE])
            # width_single.set_shape([ONE])

            # image_shape = tf.shape(image_single)
            # height_shape = tf.shape(height_single)
            # width_shape = tf.shape(width_single)
            # depth_shape = tf.shape(depth_single)
            # box_shape = tf.shape(bbox_single)
            # class_shape = tf.shape(class_name_single)
            # filename_shape = tf.shape(file_name_single)
            # label_shape = tf.shape(label_single)
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     print(sess.run(image_shape))
            #     print(sess.run(height_shape))
            #     print(sess.run(width_shape))
            #     print(sess.run(depth_shape))
            #     print(sess.run(box_shape))
            #     print(sess.run(class_shape))
            #     print(sess.run(filename_shape))
            #     print(sess.run(label_shape))


            return self._read_and_decode_astronet(filename_queue)
            # Shuffle the examples and collect them into batch_size batches.
            '''
            tf.train.shuffle_batch needs to know all the shapes of the tensors, which are presumed to be a fixed size.
            However, the bounded box tensor shape is [N,4], where N is the number of object detections for this image.
            The N varies from image to image and is not a fixed number. So, tf.train.shuffle_batch can't really handle this scenario.
            See: https://stackoverflow.com/questions/39456554/tensorflow-tensor-with-inconsistent-dimension-size for more details.
            SOLUTION:
            One way around this is to store the image AND annotation files as tfrecords. In this case, tf.train.shuffle_batch would work just fine.
            The annotation file bounded box information would then have to be decoded after the batch shuffle.
            For now, let's kick the can down the road and not implement tr.train.shuffle_batch. That is, the train, validation and test tfrecord file inputs
            were already randomized in order. So, this should be okay for now.
            '''

            # return tf.train.shuffle_batch(
            #         [image_single, height_single, width_single, depth_single, label_single, class_name_single, file_name_single, bbox_single],
            #         batch_size=batch_size,
            #         capacity=capacity,
            #         num_threads=num_threads,
            #         min_after_dequeue=min_after_dequeue)
            #         # shapes=[[515,512,3],[1],[1],[1],[1],[1,],[1,],[1,4]])

    def get_val_batch_ops(self,
                          batch_size=11429,
                          capacity=1429,
                          num_threads=16,
                          min_after_dequeue=100):

        print('In get_val_batch_ops:')
        
        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.val_file)

        # Create an input scope for the graph.
        with tf.name_scope('val_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename], capacity=1)

            return self._read_and_decode_astronet(filename_queue)

    def get_test_batch_ops(self,
                          batch_size=1428,
                          capacity=1428,
                          num_threads=16,
                          min_after_dequeue=100):

        print('In get_test_batch_ops:')

        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.test_file)

        # Create an input scope for the graph.
        with tf.name_scope('val_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename], capacity=1)

            return self._read_and_decode_astronet(filename_queue)

    def test_batch_read_from_records(self, 
                                     which_batch, 
                                     path_to_save_files,  
                                     numBatches=2):
        if which_batch == 'train':

            image_batch,\
            height_batch,\
            width_batch,\
            depth_batch,\
            label_batch,\
            class_name_batch,\
            file_name_batch,\
            bbox_batch = self.get_train_batch_ops(batch_size=numBatches,
                                                  capacity=33,
                                                  num_threads=2,
                                                  min_after_dequeue=10)
        elif which_batch == 'valid':

            image_batch,\
            height_batch,\
            width_batch,\
            depth_batch,\
            label_batch,\
            class_name_batch,\
            file_name_batch,\
            bbox_batch = self.get_val_batch_ops(batch_size=numBatches,
                                                capacity=32,
                                                num_threads=2,
                                                min_after_dequeue=10)
        elif which_batch == 'test':

            image_batch,\
            height_batch,\
            width_batch,\
            depth_batch,\
            label_batch,\
            class_name_batch,\
            file_name_batch,\
            bbox_batch = self.get_test_batch_ops(batch_size=numBatches,
                                                 capacity=32,
                                                 num_threads=2,
                                                 min_after_dequeue=10)

        else:
            print('which_batch not valid')
            sys.exit()

        # TODO: max_num_detections =3. Change if > 3 if required
        class_detectons_batch = tf.cast([1,1,1],tf.int32) 
        scores_batch = tf.cast([1.,1.,1.], tf.float32)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session()  as sess:
        
            sess.run(init_op)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            category_index = {1: {'name': 'satellite'}}

            # Read off numBatches batches
            for i in range(numBatches):
            
                images,\
                heights,\
                widths,\
                depths,\
                category_indices,\
                class_detectons,\
                file_names,\
                boxes,\
                scores = sess.run([ image_batch, 
                                    height_batch, 
                                    width_batch, 
                                    depth_batch, 
                                    label_batch, 
                                    class_detectons_batch, 
                                    file_name_batch, 
                                    bbox_batch, 
                                    scores_batch])
 
                file_names = file_names.decode("utf-8")
                output_path = os.path.join(path_to_save_files, file_names)

                print('images.shape:', images.shape)
                print('heights:', heights)
                print('widths:', widths)
                print('depths:', depths)
                print('boxes.shape:', boxes.shape)
                print('boxes:',boxes)
                print('class_detectons:', class_detectons)
                print('scores:',scores)
                print('category_indices:',category_index)
                print('category_index:', category_index)
                print('file_names:', file_names)

                visualize_boxes_and_labels_on_image_array(
                      images,
                      boxes,
                      class_detectons,
                      scores=scores,
                      category_index=category_index,
                      instance_masks=None,
                      use_normalized_coordinates=True,
                      line_thickness=2)

                save_image_array_as_png(images, output_path)
            
            coord.request_stop()
            coord.join(threads)
            print('')

def main(unused_argv):

    data_dir = os.path.expanduser(FLAGS.directory)

    train_file = FLAGS.train_tfrecords
    test_file = FLAGS.test_tfrecords
    valid_file = FLAGS.valid_tfrecords

    batch_producer = AstroNetBatchProducer(data_dir=data_dir,
                                         train_file=train_file,
                                         val_file=test_file,
                                         test_file=valid_file)

    train_path_to_save_files = os.path.join(data_dir, FLAGS.train_batch_out)
    test_path_to_save_files = os.path.join(data_dir, FLAGS.test_batch_out)
    valid_path_to_save_files = os.path.join(data_dir, FLAGS.valid_batch_out) 

    batch_producer.test_batch_read_from_records(which_batch='train', 
                                                path_to_save_files=train_path_to_save_files,
                                                numBatches=32)
    batch_producer.test_batch_read_from_records(which_batch='valid', 
                                                path_to_save_files=valid_path_to_save_files,
                                                numBatches=16)
    batch_producer.test_batch_read_from_records(which_batch='test', 
                                                path_to_save_files=test_path_to_save_files,
                                                numBatches=8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str, 
                        default='~/tensorflow/false_positives/data', 
                        help='Directory where to save the batch images')
    parser.add_argument('--train_tfrecords',
                        type=str, 
                        default='train.tfrecords', 
                        help='train.tfrecords input file')
    parser.add_argument('--test_tfrecords',
                        type=str, 
                        default='test.tfrecords', 
                        help='test.tfrecords input file')
    parser.add_argument('--valid_tfrecords',
                        type=str, 
                        default='valid.tfrecords', 
                        help='valid.tfrecords input file')
    parser.add_argument('--train_batch_out',
                        type=str, 
                        default='test_batch_producer/train', 
                        help='partial path to batch producer train output')
    parser.add_argument('--test_batch_out',
                        type=str, 
                        default='test_batch_producer/test', 
                        help='partial path to batch producer test output')
    parser.add_argument('--valid_batch_out',
                        type=str, 
                        default='test_batch_producer/valid', 
                        help='partial path to batch producer valid output')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


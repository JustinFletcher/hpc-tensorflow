
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

MAX_NUM_DETECTIONS = 10
NO_BBOX_DETECTION = [-1. -1. -1. -1]

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
                # 'image/depth': tf.FixedLenFeature([], tf.int64, 1),
                'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/source_id': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/key/sha256': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/format': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                'image/object/class/text': tf.VarLenFeature(tf.string),
                'image/object/class/label': tf.VarLenFeature(tf.int64)
            }


    def _read_and_decode_astronet(self, filename_queue):

        # print('In _read_and_decode_astronet():')
        # Instantiate a TFRecord reader.
        reader = tf.TFRecordReader()

        # Read a single example from the input queue.
        # print('reader.read():')
        _, serialized_example = reader.read(filename_queue)

        # Parse the example into features.
        # print('tf.parse_single_example:')
        features = tf.parse_single_example(serialized_example,self.keys_to_features)

        # Convert from a scalar string tensor to a uint8 tensor with shape [height,width,channels].
        # print('tf.image.decode_jpeg:')
        image = tf.image.decode_jpeg(features['image/encoded'],channels=3)

        height = tf.cast(features['image/height'], tf.int32)
        width = tf.cast(features['image/width'], tf.int32)
        # depth = tf.cast(features['image/depth'], tf.int32)
    
        # print('height',height)
        # print('width',width)
        # print('depth',depth)
        # image_shape = tf.stack([height, width, depth])       
        # image = tf.reshape(image, image_shape)
        image = tf.reshape(image, [512,512,3])
        # print('image.shape',image.shape)

        label = tf.cast(features['image/object/class/label'], tf.int32)
        class_name = tf.cast(features['image/object/class/text'], tf.string)
        file_name = tf.cast(features['image/filename'], tf.string)

        # print('BoundingBox():')
        bbox = BoundingBox(keys=None, prefix='image/object/bbox/').tensors_to_item(features)

        # pad the number of bounding boxes to a fixed size
        paddings = [[0, MAX_NUM_DETECTIONS-tf.shape(bbox)[0]], [0, 0]]
        # print('bbox:',bbox)
        # print('tf.shape(bbox)[0]:',tf.shape(bbox)[0])
        # print('paddings:',paddings)
        # print('bbox_padded:')
        bbox_padded = tf.pad(bbox, paddings, 'CONSTANT', constant_values=-1.0)

        bbox = tf.reshape(bbox_padded,(10,4))

        print('return:')
        return image, height, width, label, class_name, file_name, bbox

    def get_train_batch_ops(self, capacity):

        print('')
        print('In get_train_batch_ops:')
        
        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.train_file)

        # Create an input scope for the graph.
        with tf.name_scope('train_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename], capacity=capacity)

            return self._read_and_decode_astronet(filename_queue)

    def get_val_batch_ops(self,capacity):

        print('')
        print('In get_val_batch_ops:')
        
        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.val_file)

        print('filename:',filename)

        # Create an input scope for the graph.
        with tf.name_scope('val_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename], capacity=capacity)

            return self._read_and_decode_astronet(filename_queue)

    def get_test_batch_ops(self,capacity):

        print('')
        print('In get_test_batch_ops:')

        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.test_file)

        # Create an input scope for the graph.
        with tf.name_scope('val_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename], capacity=capacity)

            return self._read_and_decode_astronet(filename_queue)

    def test_batch_read_from_records(self, 
                                     which_batch, 
                                     path_to_save_files,  
                                     numBatches=2,
                                     batch_size = 2,
                                     capacity = 32,
                                     num_threads = 2,
                                     ):
        if which_batch == 'train':
            
            image,\
            height,\
            width,\
            label,\
            class_name,\
            file_name,\
            bbox = self.get_train_batch_ops(capacity=32)

        elif which_batch == 'valid':

            image,\
            height,\
            width,\
            label,\
            class_name,\
            file_name,\
            bbox = self.get_val_batch_ops(capacity=32)

        elif which_batch == 'test':

            image,\
            height,\
            width,\
            label,\
            class_name,\
            file_name,\
            bbox = self.get_test_batch_ops(capacity=32)

        else:
            print('which_batch not valid')
            sys.exit()

        class_detection = tf.cast([1]*MAX_NUM_DETECTIONS,tf.int32) 
        score = tf.cast([1.]*MAX_NUM_DETECTIONS, tf.float32)

        print('In tf.train.batch():')
        image_batch,\
        height_batch,\
        width_batch,\
        label_batch,\
        class_name_batch,\
        file_name_batch,\
        bbox_batch,\
        class_detection_batch,\
        score_batch = tf.train.batch([image, height, width, label, class_name, file_name, bbox, class_detection, score], 
            batch_size=batch_size,
            num_threads = num_threads,
            capacity = capacity,
            allow_smaller_final_batch = True)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
        
            sess.run(init_op)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            category_index = {1: {'name': 'satellite'}}

            print('Starting batch processing:')
            # Read off numBatches batches
            # for i in range(numBatches):
            count = 0

            try:
                # in most cases coord.should_stop() will return True
                # when there are no more samples to read
                # if num_epochs=0 then it will run for ever
                while not coord.should_stop():
                        # will start reading, working data from input queue
                        # and "fetch" the results of the computation graph
                        # into raw_images and other raw data
                    images,\
                    heights,\
                    widths,\
                    labels,\
                    class_names,\
                    file_names,\
                    bboxes,\
                    class_detections,\
                    scores = sess.run([image_batch, height_batch, width_batch, label_batch, class_name_batch, file_name_batch, bbox_batch, class_detection_batch, score_batch])
     
                    file_names   = [file_name.decode("utf-8") for file_name in file_names]
                    output_paths = [os.path.join(path_to_save_files, file_name) for file_name in file_names]

                    print('')
                    print('images.shape:', images.shape)
                    print('heights:', heights)
                    print('widths:', widths)
                    print('bboxes.shape:', bboxes.shape)
                    print('bboxes:')
                    print(bboxes)
                    print('class_detections:')
                    print(class_detections)
                    print('class_detections.shape:')
                    print(class_detections.shape)
                    print('scores:')
                    print(scores)
                    print('scores.shape:')
                    print(scores.shape)
                    print('category_index:', category_index)
                    print('file_names:')
                    print(file_names)
                    print('output_paths:')
                    print(output_paths)

                    # process each image in the batch:
                    for i,output_path in enumerate(output_paths):

                        # note: these are numpy arrays and not lists
                        image = images[i,:,:,:]
                        boxes = bboxes[i,:,:]
                        # find out how many real detections:
                        boxes = boxes[np.logical_not(boxes[:,0] < 0)]
                        num_bboxes = boxes.shape[0]
                        class_detection = class_detections[i].flatten()
                        class_detection = class_detection[:num_bboxes]
                        score = scores[i].flatten()
                        score = score[:num_bboxes]

                        print('boxes:')
                        print(boxes)
                        print('num_bboxes:',num_bboxes)
                        print('boxes.shape:',boxes.shape)
                        print('class_detection:')
                        print(class_detection)
                        print('score:')
                        print(score)

                        visualize_boxes_and_labels_on_image_array(
                              image,
                              boxes,
                              class_detection,
                              scores=score,
                              category_index=category_index,
                              instance_masks=None,
                              use_normalized_coordinates=True,
                              line_thickness=2)

                        save_image_array_as_png(image, output_path)

                    count += 1

                    if count >= numBatches:
                        break
            
            finally:
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

    # batch_producer.test_batch_read_from_records(which_batch='train', 
    #                                             path_to_save_files=train_path_to_save_files,
    #                                             numBatches=32,
    #                                             batch_size =2)
    batch_producer.test_batch_read_from_records(which_batch='valid', 
                                                path_to_save_files=valid_path_to_save_files,
                                                numBatches=3,
                                                batch_size =5)
    # batch_producer.test_batch_read_from_records(which_batch='test', 
    #                                             path_to_save_files=test_path_to_save_files,
    #                                             numBatches=8
    #                                             batch_size =2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str, 
                        default='~/tensorflow/false_positives/datat', 
                        help='Directory where to save the batch images')
    parser.add_argument('--train_tfrecords',
                        type=str, 
                        default='astronet_train.tfrecords',
                        help='astronet_train.tfrecords input file')
    parser.add_argument('--test_tfrecords',
                        type=str, 
                        default='astronet_test.tfrecords',
                        help='astronet_test.tfrecords input file')
    parser.add_argument('--valid_tfrecords',
                        type=str, 
                        default='astronet_valid.tfrecords',
                        help='astronet_valid.tfrecords input file')
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


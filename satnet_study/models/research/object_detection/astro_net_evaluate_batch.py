
"""Evaluate atro_net training model by batch processing images from tf_records.

Example usage:
qsub -I -A MHPCC96650DE1 -q standard -l select=1:ncpus=20:mpiprocs=20 -l walltime=1:00:00
module load anaconda3 tensorflow
module unload anaconda2
cd /gpfs/projects/ml/tfmodels/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo $PYTHONPATH
cd /gpfs/projects/ml/hpc-tensorflow/satnet_study/models/research
python object_detection/astro_net_evaluate_batch.py
# note: print messages not showing up in terminal
ls | wc -l

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import tensorflow as tf
# import zipfile
import pickle

from tensorflow.contrib.slim.python.slim.data.tfexample_decoder import BoundingBox
from utils import label_map_util
from utils.visualization_utils import save_image_array_as_png 
from utils.visualization_utils import visualize_boxes_and_labels_on_image_array 

sys.path.append("..")

FLAGS = None

NUM_CLASSES = 1
MAX_NUM_DETECTIONS = 10
NO_BBOX_DETECTION = [-1. -1. -1. -1]

class AstroNetEvaluateBatch:

    def __init__(self,
                 path_to_frozen_graph,
                 path_to_labels_map):

        self.detection_graph = self.load_graph(path_to_frozen_graph)
        self.category_index  = self.get_category_index(path_to_labels_map)
        self.tensor_dict     = {}
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
        self.get_tensor_dict_from_graph()
        # self.image_tensor = self.get_image_tensor_from_graph()
        self.detections_dict = {}
        self.detections_dict['image_name'] = []
        self.detections_dict['num_detections'] = []
        self.detections_dict['detection_classes'] = []
        self.detections_dict['detection_scores'] = []
        self.detections_dict['detection_boxes'] = []
        self.detections_dict['ground_truth_class_id'] = []
        self.detections_dict['ground_truth_boxes'] = []

    def load_graph(self,path_to_frozen_graph):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        # print('Loading graph:')
        with tf.gfile.GFile(path_to_frozen_graph, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Import the graph_def into a new Graph and return it 
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph

    def get_category_index(self,path_to_labels_map):
        # print('get category_index:')
        label_map      = label_map_util.load_labelmap(path_to_labels_map)
        categories     = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def get_tensor_dict_from_graph(self):
        # print('get_tensor_dict_from_graph:')
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                for key in [
                      'num_detections', 'detection_boxes', 'detection_scores',
                      'detection_classes'
                    ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    def saveAsPickle(self,pickle_file):
        # Store data (serialize)
        print('Saving pickle file...')
        with open(pickle_file, 'wb') as handle:
            pickle.dump(self.detections_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

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

        # print('return:')
        return image, height, width, label, class_name, file_name, bbox

    def process_tfrecords(self, 
                          path_to_tfrecords,
                          data_out_dir,
                          batch_size,
                          numBatches,
                          capacity,
                          num_threads,
                         ):
 

        image_count = 0
        with self.detection_graph.as_default():
            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([path_to_tfrecords], capacity=capacity, num_epochs=1)

            # read and decode the tf_example:
            image,\
            height,\
            width,\
            label,\
            class_name,\
            file_name,\
            bbox = self._read_and_decode_astronet(filename_queue)

            class_detection = tf.cast([1]*MAX_NUM_DETECTIONS,tf.int32) 
            score = tf.cast([1.]*MAX_NUM_DETECTIONS, tf.float32)

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

            with tf.Session() as sess:
            
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)
                
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)

                print('Starting batch processing:')

                batch_count = 0

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
                        output_paths = [os.path.join(data_out_dir, file_name) for file_name in file_names]

                        # print('')
                        # print('images.shape:', images.shape)
                        # print('heights:', heights)
                        # print('widths:', widths)
                        # print('bboxes.shape:', bboxes.shape)
                        # print('bboxes:')
                        # print(bboxes)
                        # print('class_detections:')
                        # print(class_detections)
                        # print('class_detections.shape:')
                        # print(class_detections.shape)
                        # print('scores:')
                        # print(scores)
                        # print('scores.shape:')
                        # print(scores.shape)
                        # print('category_index:', self.category_index)
                        # print('file_names:')
                        # print(file_names)
                        # print('output_paths:')
                        # print(output_paths)

                        # Run inference
                        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                        output_dict = sess.run(self.tensor_dict, feed_dict={image_tensor: images})

                        # all outputs are float32 numpy arrays, so convert types as appropriate
                        output_dict['num_detections'] = output_dict['num_detections'].astype(np.uint8)
                        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.uint8)

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

                            # print(file_names[i])
                            # print('boxes:')
                            # print(boxes)
                            # print('num_bboxes:',num_bboxes)
                            # print('boxes.shape:',boxes.shape)
                            # print('class_detection:')
                            # print(class_detection)
                            # print('score:')
                            # print(score)

                            self.detections_dict['image_name'].append(file_names[i])
                            self.detections_dict['num_detections'].append(output_dict['num_detections'][i])
                            self.detections_dict['detection_classes'].append(output_dict['detection_classes'][i].tolist())
                            self.detections_dict['detection_scores'].append(output_dict['detection_scores'][i].tolist())
                            self.detections_dict['detection_boxes'].append(output_dict['detection_boxes'][i].tolist())
                            self.detections_dict['ground_truth_class_id'].append(class_detection.tolist())
                            self.detections_dict['ground_truth_boxes'].append(boxes.tolist())

                            # Call this routine twice to plot: 1) detections; 2) ground truth
                            visualize_boxes_and_labels_on_image_array(
                                  image,
                                  output_dict['detection_boxes'][i],
                                  output_dict['detection_classes'][i],
                                  scores=output_dict['detection_scores'][i],
                                  category_index=self.category_index,
                                  instance_masks=None,
                                  use_normalized_coordinates=True,
                                  line_thickness=2)

                            visualize_boxes_and_labels_on_image_array(
                                  image,
                                  boxes,
                                  class_detection,
                                  scores=None,
                                  category_index=self.category_index,
                                  instance_masks=None,
                                  use_normalized_coordinates=True,
                                  line_thickness=2,
                                  groundtruth_box_visualization_color='white')

                            save_image_array_as_png(image, output_path)

                            image_count += 1

                            print(image_count,file_names[i])

                        batch_count += 1

                        if numBatches is not None and batch_count >= numBatches:
                            break

                except Exception as e:
                    coord.request_stop(e)
                    coord.join(threads)

                finally:
                    coord.request_stop()
                    coord.join(threads)

        # save data for mAP analysis:
        pickle_file = os.path.join(data_out_dir,'detections.pickle')
        self.saveAsPickle(pickle_file)

        print('image_count:', image_count)

def main(unused_argv):

    # data_in_dir  = os.path.expanduser(FLAGS.directory_in)
    # data_out_dir = os.path.expanduser(FLAGS.directory_out)
    data_out_dir = FLAGS.directory_out

    # tf_records_filename = os.path.expanduser(FLAGS.filename_in_tfrecords)

    path_to_checkpoint = FLAGS.checkpoint_path

    path_to_labels = FLAGS.labels_path


    astro_evaluator = AstroNetEvaluateBatch(path_to_frozen_graph=path_to_checkpoint,
                                            path_to_labels_map = path_to_labels)

    # path_to_tfrecords = os.path.join(data_in_dir, tf_records_filename)
    path_to_tfrecords = FLAGS.filename_in_tfrecords

    # print('path_to_tfrecords:',path_to_tfrecords)

    astro_evaluator.process_tfrecords(path_to_tfrecords,
                                      data_out_dir,
                                      FLAGS.batch_size,
                                      FLAGS.numBatches,
                                      FLAGS.capacity,
                                      FLAGS.num_threads)

    print('main: Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_in',
                        type=str, 
                        default='/gpfs/projects/ml/data/satdetect', 
                        help='Directory to input data')
    parser.add_argument('--filename_in_tfrecords',
                        type=str, 
                        default='/gpfs/projects/ml/data/satdetect/astronet_test_3.tfrecords',
                        help='tfrecords input file')
    parser.add_argument('--directory_out',
                        type=str, 
                        default='/gpfs/projects/ml/data/satdetect/inference/trial_3/test_images_out_3',
                        help='Directory where to save output data')
    parser.add_argument('--checkpoint_path',
                        type=str, 
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_study/models/research/exported_graphs/train-3/frozen_inference_graph.pb',
                        help='Path to exported graph model checkpoint')
    parser.add_argument('--labels_path',
                        type=str, 
                        default='/gpfs/projects/ml/data/satdetect/astronet_label_map_2.pbtxt',
                        help='Path to labels map')
    parser.add_argument('--batch_size',
                        type=int, 
                        default=8,
                        help='Number of images to process per batch')
    parser.add_argument('--numBatches',
                        type=int, 
                        default=None,
                        help='Number of batches to process. If None, process the entire tf_record.')
    parser.add_argument('--capacity',
                        type=int, 
                        default=32,
                        help='Queue depth (capacity)')
    parser.add_argument('--num_threads',
                        type=int, 
                        default=2,
                        help='number of threads')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


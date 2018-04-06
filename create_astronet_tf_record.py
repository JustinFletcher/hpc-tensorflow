# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Converts astronet image and annotation data to TFRecords file format with Example protos.

Example usage:
    python3 create_astronet_tf_record.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import io
import logging
import os
import sys
import glob
import numpy as np

from PIL import Image
# import skimage.io

import tensorflow as tf

sys.path.append("../../") # Adds higher directory to python modules path.

from object_detection.utils import dataset_util

FLAGS = None

CLASS_TEXT = ['None','Satellite']

def read_annotation_file(filename):
    """Reads an annotation file.

    Converts an astronet annotation file into a dictionary containing all the
    relevant information.

    Args:
    filename: the path to the annotataion text file.

    Returns:
    anno: A dictionary with the converted annotation information. 
    """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]

    # fix for label: index is off by 1
    label = [int(x[0])+1 for x in content]
    # print('label:',label)

    x_center = [float(x[1]) for x in content]
    y_center = [float(x[2]) for x in content]
    bbox_width = [float(x[3]) for x in content]
    bbox_height = [float(x[4]) for x in content]

    # clip ranges to 0 and 1
    y_min = [max(y0-h/2,0.) for y0, h in zip(y_center, bbox_height)]
    y_max = [min(y0+h/2,1.) for y0, h in zip(y_center, bbox_height)]

    x_min = [max(x0-w/2,0.) for x0, w in zip(x_center, bbox_width)]
    x_max = [min(x0+w/2,1.) for x0, w in zip(x_center, bbox_width)]

    anno = {}
    anno['class_id'] = np.array(label)

    anno['y_min'] = np.array(y_min)
    anno['y_max'] = np.array(y_max)

    anno['x_min'] = np.array(x_min)
    anno['x_max'] = np.array(x_max)

    return anno

def convert_astronet_to_tfrecords(path_to_txt, path_to_tfrecords, verify = True):
    """Converts an astronet dataset to tfrecords."""

    # generate the image_path_list:
    with open(path_to_txt) as fptr:
        image_str = fptr.read()

    image_path_list = image_str.split('\n')
    # remove the last '' element from the list
    image_path_list = image_path_list[:-1] 

    annot_str = image_str.replace('.jpg', '.txt')
    annot_path_list = annot_str.split('\n')
    # remove the last '' element from the list
    annot_path_list = annot_path_list[:-1]

    num_examples = len(image_path_list)

    # Build a writer for the tfrecord.
    print('Writing to', path_to_tfrecords, '... Num_records = ', num_examples)
    writer = tf.python_io.TFRecordWriter(path_to_tfrecords)

    # Iterate over the examples.
    for index in range(num_examples):
        # image = np.array(Image.open(image_path_list[index]))

        with tf.gfile.GFile(image_path_list[index], 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()
        # convert to a numpy array:
        image = np.array(image)
        h = int(image.shape[0])
        w = int(image.shape[1])
        c = int(image.shape[2])
        #check if RGBA format and convert to RGB if true:
        if c == 4:
            # print('Image is RGBA. Convert to RGB.')
            raise ValueError('Image is RGBA')
        #     image = image[:,:,:3]
        #     c = 3
        # image_raw = image.tostring()

        annotations = read_annotation_file(annot_path_list[index])
        # Note: annotation data is already normalized:
        ymin_norm = annotations['y_min']
        ymax_norm = annotations['y_max']

        xmin_norm = annotations['x_min']
        xmax_norm = annotations['x_max']

        filename = image_path_list[index].split('/')[-1]
        # print('filename',filename)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(h),
            'image/width': dataset_util.int64_feature(w),
            'image/depth': dataset_util.int64_feature(c),
            'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
            'image/object/class/text': dataset_util.bytes_list_feature([CLASS_TEXT[x].encode('utf8') for x in annotations['class_id']]),
            'image/object/class/label': dataset_util.int64_list_feature([x for x in annotations['class_id']])
            }))
        writer.write(example.SerializeToString())
        # break  # debug
    writer.close()

    if verify:
        print('Verifying...', path_to_tfrecords)
        record_iterator = tf.python_io.tf_record_iterator(path=path_to_tfrecords)

        index = 0
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            height_recon = int(example.features.feature['image/height']
                                 .int64_list
                                 .value[0])

            width_recon = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])

            depth_recon = int(example.features.feature['image/depth']
                                .int64_list
                                .value[0])

            filename_recon = example.features.feature['image/filename'].bytes_list.value[0].decode('utf-8')
            # print('filename_recon',filename_recon)

            encoded_jpg_recon = (example.features.feature['image/encoded']
                                  .bytes_list
                                  .value[0])

            encoded_jpg_io_recon  = io.BytesIO(encoded_jpg_recon)
            image_recon = Image.open(encoded_jpg_io_recon)
            # img_1d = np.fromstring(img_string, dtype=np.uint8)
            # image_recon = img_1d.reshape((height_recon, width_recon, depth_recon))

            ymin_recon = np.array(example.features.feature['image/object/bbox/ymin'].float_list.value)
            ymax_recon = np.array(example.features.feature['image/object/bbox/ymax'].float_list.value)
            xmin_recon = np.array(example.features.feature['image/object/bbox/xmin'].float_list.value)
            xmax_recon = np.array(example.features.feature['image/object/bbox/xmax'].float_list.value)

            class_label_recon = np.array(example.features.feature['image/object/class/label'].int64_list.value)

            # get origiinal image, annot:
            # image = np.array(Image.open(image_path_list[index]))

            with tf.gfile.GFile(image_path_list[index], 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()
            # convert to a numpy array:
            image = np.array(image)
            height = int(image.shape[0])
            width = int(image.shape[1])
            depth = int(image.shape[2])
            #check if RGBA format and convert to RGB if true:
            if depth == 4:
                # print('Image is RGBA. Convert to RGB.')
                raise ValueError('Image is RGBA')
                # image = image[:,:,:3]
                # depth = 3

            filename = image_path_list[index].split('/')[-1]

            annotations = read_annotation_file(annot_path_list[index])

            # Note: annotation data is already normalized:
            ymin = np.array(annotations['y_min'])
            ymax = np.array(annotations['y_max'])

            xmin = np.array(annotations['x_min'])
            xmax = np.array(annotations['x_max'])

            class_label = np.array(annotations['class_id'])

            #check if equal
            if not np.allclose(image,image_recon):
                print('images are not equal for index:',index)

            if not np.allclose(height,height_recon):
                print('image height is not equal for index:',index, height, height_recon)

            if not np.allclose(width,width_recon):
                print('image width is not equal for index:',index, width, width_recon)

            if not np.allclose(depth,depth_recon):
                print('image depth is not equal for index:',index, depth, depth_recon)

            if not filename == filename_recon:
                print('filename is not equal for index:',index, filename, filename_recon, len(filename), len(filename_recon))

            # print('xmin,xmin_recon',xmin,xmin_recon)
            if not np.allclose(xmin,xmin_recon):
                print('image bbox xmin is not equal for index:',index, xmin, xmin_recon)

            if not np.allclose(ymin,ymin_recon):
                print('image bbox ymin is not equal for index:',index, ymin, ymin_recon)

            if not np.allclose(xmax,xmax_recon):
                print('image bbox xmax is not equal for index:',index, xmax, xmax_recon)

            if not np.allclose(ymax,ymax_recon):
                print('image bbox ymax is not equal for index:',index, ymax, ymax_recon)

            if not np.allclose(class_label,class_label_recon):
                print('image bbox class_label is not equal for index:',index, class_label, class_label_recon)

            index += 1
            
            # break  # debug

    print('Done')


def main(unused_argv):

    sourceDir = os.path.expanduser(FLAGS.directory)

    path_to_train_txt = os.path.join(sourceDir, 'train.txt')
    path_to_test_txt = os.path.join(sourceDir, 'test.txt')
    path_to_valid_txt = os.path.join(sourceDir, 'valid.txt')

    path_to_train_tfrecords = os.path.join(sourceDir, 'astronet_train.tfrecords')
    path_to_test_tfrecords = os.path.join(sourceDir, 'astronet_test.tfrecords')
    path_to_valid_tfrecords = os.path.join(sourceDir, 'astronet_valid.tfrecords')

    print('Generating train.tfrecords...')
    convert_astronet_to_tfrecords(
        path_to_txt = path_to_train_txt, 
        path_to_tfrecords = path_to_train_tfrecords)

    print('Generating test.tfrecords...')
    convert_astronet_to_tfrecords(
        path_to_txt = path_to_test_txt, 
        path_to_tfrecords = path_to_test_tfrecords)

    print('Generating valid.tfrecords...')
    convert_astronet_to_tfrecords(
        path_to_txt = path_to_valid_txt, 
        path_to_tfrecords = path_to_valid_tfrecords)

    print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str, 
                        default='~/tensorflow/false_positives/datat', 
                        help='Directory (train,test,valid).txt files and where to store corresponding *.tfrecords')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

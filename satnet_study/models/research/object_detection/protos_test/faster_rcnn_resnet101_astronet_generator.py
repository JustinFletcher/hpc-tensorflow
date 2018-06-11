#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Generate faster_rcnn_resnet101_astronet.config file and permit varying certain paramters.

from google.protobuf import text_format
import faster_rcnn_resnet101_astronet_pb2
from object_detection.protos import model_pb2
from object_detection.protos import faster_rcnn_pb2
from object_detection.protos import train_pb2
from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import optimizer_pb2
from object_detection.protos import image_resizer_pb2
from object_detection.protos import anchor_generator_pb2
from object_detection.protos import grid_anchor_generator_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.protos import box_predictor_pb2
from object_detection.protos import post_processing_pb2

import sys
import argparse
import os

class AstroNetFasterRcnnResnet101Generator:
    def __init__(self,
                 config_file_name):
        self.config_file_name = config_file_name

        self.faster_rcnn_obj = FasterRcnn()
        self.detection_model_obj = DetectionModel(self.faster_rcnn_obj.faster_rcnn)
        self.train_config_obj = TrainConfig()
        self.eval_config_obj = EvalConfig()

        self.train_tfrecord_input_reader_obj = TrainTFRecordInputReader()
        self.eval_tfrecord_input_reader_obj = EvalTFRecordInputReader()

        self.train_input_reader_obj = TrainInputReader(self.train_tfrecord_input_reader_obj.tf_record_input_reader)
        self.eval_input_reader_obj = EvalInputReader(self.eval_tfrecord_input_reader_obj.tf_record_input_reader)

        self.faster_rcnn_astronet = faster_rcnn_resnet101_astronet_pb2.FasterRCNNResnet101AstroNet()

        self.faster_rcnn_astronet.model.CopyFrom(self.detection_model_obj.detection_model)
        self.faster_rcnn_astronet.train_config.CopyFrom(self.train_config_obj.train_config)
        self.faster_rcnn_astronet.eval_config.CopyFrom(self.eval_config_obj.eval_config)
        self.faster_rcnn_astronet.train_input_reader.CopyFrom(self.train_input_reader_obj.input_reader)
        self.faster_rcnn_astronet.eval_input_reader.CopyFrom(self.eval_input_reader_obj.input_reader)

    def batch_size(self, batch_size):
        self.faster_rcnn_astronet.train_config.batch_size = batch_size

    def initial_learning_rate(self, initial_learning_rate):
        self.faster_rcnn_astronet.train_config.optimizer.adam_optimizer.learning_rate.exponential_decay_learning_rate.initial_learning_rate = initial_learning_rate

    def config_file_in(config_file_name):
        self.config_file_name = config_file_name

    def config_file_output(self):
        print('config_file:',self.config_file_name)
        # Write the model back to disk.
        with open(self.config_file_name, "w") as f:
          f.write(text_format.MessageToString(self.faster_rcnn_astronet))        

class FasterRcnn:
    def __init__(self):
        self.faster_rcnn = faster_rcnn_pb2.FasterRcnn()
        self.image_resizer = image_resizer_pb2.ImageResizer()
        self.fixed_shape_resizer = image_resizer_pb2.FixedShapeResizer()
        self.feature_extractor = faster_rcnn_pb2.FasterRcnnFeatureExtractor()
        self.anchor_generator = anchor_generator_pb2.AnchorGenerator()
        self.grid_anchor_generator = grid_anchor_generator_pb2.GridAnchorGenerator()
        self.hyperparams_1st_stage = hyperparams_pb2.Hyperparams()
        self.hyperparams_2nd_stage = hyperparams_pb2.Hyperparams()
        self.regularizer_1st_stage = hyperparams_pb2.Regularizer()
        self.regularizer_2nd_stage = hyperparams_pb2.Regularizer()
        self.l2_regularizer_1st_stage = hyperparams_pb2.L2Regularizer()
        self.l2_regularizer_2nd_stage = hyperparams_pb2.L2Regularizer()
        self.initializer_1st_stage = hyperparams_pb2.Initializer()
        self.initializer_2nd_stage = hyperparams_pb2.Initializer()
        self.truncated_normal_initializer = hyperparams_pb2.TruncatedNormalInitializer()
        self.variance_scaling_initializer = hyperparams_pb2.VarianceScalingInitializer()
        self.box_predictor = box_predictor_pb2.BoxPredictor()
        self.mask_rcnn_box_predictor = box_predictor_pb2.MaskRCNNBoxPredictor()
        self.post_processing = post_processing_pb2.PostProcessing()
        self.batch_non_max_suppression = post_processing_pb2.BatchNonMaxSuppression()

        self.fixed_shape_resizer.height = 512
        self.fixed_shape_resizer.width = 512

        self.image_resizer.fixed_shape_resizer.CopyFrom(self.fixed_shape_resizer)

        self.feature_extractor.type = 'faster_rcnn_resnet101'
        self.feature_extractor.first_stage_features_stride = 16

        self.grid_anchor_generator.height_stride = 16
        self.grid_anchor_generator.width_stride = 16
        self.grid_anchor_generator.scales[:] = [0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 1.0, 2.0]
        self.grid_anchor_generator.aspect_ratios.append(1.0) 

        self.anchor_generator.grid_anchor_generator.CopyFrom(self.grid_anchor_generator)

        self.hyperparams_1st_stage.op = hyperparams_pb2.Hyperparams.CONV
        self.hyperparams_2nd_stage.op = hyperparams_pb2.Hyperparams.FC
        self.truncated_normal_initializer.stddev = 0.01
        self.variance_scaling_initializer.factor = 1.0
        self.variance_scaling_initializer.uniform = True
        self.variance_scaling_initializer.mode = hyperparams_pb2.VarianceScalingInitializer.FAN_AVG
        self.initializer_1st_stage.truncated_normal_initializer.CopyFrom(self.truncated_normal_initializer)
        self.initializer_2nd_stage.variance_scaling_initializer.CopyFrom(self.variance_scaling_initializer)
        self.hyperparams_1st_stage.initializer.CopyFrom(self.initializer_1st_stage)
        self.hyperparams_2nd_stage.initializer.CopyFrom(self.initializer_2nd_stage)
        self.l2_regularizer_1st_stage.weight = 0.0
        self.l2_regularizer_2nd_stage.weight = 0.0
        self.regularizer_1st_stage.l2_regularizer.CopyFrom(self.l2_regularizer_1st_stage)
        self.regularizer_2nd_stage.l2_regularizer.CopyFrom(self.l2_regularizer_2nd_stage)
        self.hyperparams_1st_stage.regularizer.CopyFrom(self.regularizer_1st_stage)
        self.hyperparams_2nd_stage.regularizer.CopyFrom(self.regularizer_2nd_stage)

        self.batch_non_max_suppression.score_threshold = 0.0
        self.batch_non_max_suppression.iou_threshold = 0.6
        self.batch_non_max_suppression.max_detections_per_class = 100
        self.batch_non_max_suppression.max_total_detections = 100

        self.post_processing.batch_non_max_suppression.CopyFrom(self.batch_non_max_suppression)
        self.post_processing.score_converter = post_processing_pb2.PostProcessing.SOFTMAX

        self.faster_rcnn.num_classes = 1
        self.faster_rcnn.image_resizer.CopyFrom(self.image_resizer)
        self.faster_rcnn.feature_extractor.CopyFrom(self.feature_extractor)
        self.faster_rcnn.first_stage_anchor_generator.CopyFrom(self.anchor_generator)
        self.faster_rcnn.first_stage_box_predictor_conv_hyperparams.CopyFrom(self.hyperparams_1st_stage)

        self.faster_rcnn.first_stage_nms_score_threshold = 0.0
        self.faster_rcnn.first_stage_nms_iou_threshold = 0.7
        self.faster_rcnn.first_stage_max_proposals = 300
        self.faster_rcnn.first_stage_localization_loss_weight = 2.0
        self.faster_rcnn.first_stage_objectness_loss_weight = 1.0
        self.faster_rcnn.initial_crop_size = 14
        self.faster_rcnn.maxpool_kernel_size = 2
        self.faster_rcnn.maxpool_stride = 2

        self.mask_rcnn_box_predictor.fc_hyperparams.CopyFrom(self.hyperparams_2nd_stage)
        self.mask_rcnn_box_predictor.use_dropout = False
        self.mask_rcnn_box_predictor.dropout_keep_probability = 1.0

        self.box_predictor.mask_rcnn_box_predictor.CopyFrom(self.mask_rcnn_box_predictor)

        self.faster_rcnn.second_stage_box_predictor.CopyFrom(self.box_predictor)
        self.faster_rcnn.second_stage_post_processing.CopyFrom(self.post_processing)
        self.faster_rcnn.second_stage_localization_loss_weight = 2.0
        self.faster_rcnn.second_stage_classification_loss_weight = 1.0

class DetectionModel:
    def __init__(self, faster_rcnn):
        self.detection_model = model_pb2.DetectionModel()
        self.detection_model.faster_rcnn.CopyFrom(faster_rcnn)

class TrainConfig:
    def __init__(self):
        self.train_config = train_pb2.TrainConfig()
        self.exponential_decay_learning_rate = optimizer_pb2.ExponentialDecayLearningRate()
        self.exponential_decay_learning_rate.initial_learning_rate = 0.002

        self.learning_rate = optimizer_pb2.LearningRate()
        self.learning_rate.exponential_decay_learning_rate.CopyFrom(self.exponential_decay_learning_rate)

        self.adam_optimizer = optimizer_pb2.AdamOptimizer()
        self.adam_optimizer.learning_rate.CopyFrom(self.learning_rate)

        self.optimizer = optimizer_pb2.Optimizer()
        self.optimizer.adam_optimizer.CopyFrom(self.adam_optimizer)

        self.train_config.batch_size = 8
        self.train_config.optimizer.CopyFrom(self.optimizer)
        self.train_config.batch_queue_capacity = 8 
        self.train_config.num_batch_queue_threads = 8
        self.train_config.fine_tune_checkpoint = "/gpfs/projects/ml/astro_net/pretrained_resnet101/model.ckpt"
        self.train_config.from_detection_checkpoint = True
        self.train_config.gradient_clipping_by_norm = 10.0
        self.train_config.num_steps = 0 # indefinitely
        self.train_config.max_number_of_boxes = 100 

class EvalConfig:
    def __init__(self):
        self.eval_config = eval_pb2.EvalConfig()
        self.eval_config.num_visualizations = 20
        self.eval_config.max_evals = 0 # indefinitely
        self.eval_config.num_examples = 5000

class TFRecordInputReader:
    def __init__(self):
        self.tf_record_input_reader = input_reader_pb2.TFRecordInputReader()
    
class TrainTFRecordInputReader(TFRecordInputReader):
    def __init__(self):
        super().__init__()
        self.tf_record_input_reader.input_path.append("/gpfs/projects/ml/data/satdetect/astronet_train_3.tfrecords")

class EvalTFRecordInputReader(TFRecordInputReader):
    def __init__(self):
        super().__init__()
        self.tf_record_input_reader.input_path.append("/gpfs/projects/ml/data/satdetect/astronet_valid_3.tfrecords")

class InputReader:
    def __init__(self):
        self.input_reader = input_reader_pb2.InputReader()
    
class TrainInputReader(InputReader):
    def __init__(self,tf_record_input_reader):
        super().__init__()
        self.input_reader.tf_record_input_reader.CopyFrom(tf_record_input_reader)
        self.input_reader.label_map_path = "/gpfs/projects/ml/data/satdetect/astronet_label_map_2.pbtxt"

class EvalInputReader(InputReader):
    def __init__(self,tf_record_input_reader):
        super().__init__()
        self.input_reader.tf_record_input_reader.CopyFrom(tf_record_input_reader)
        self.input_reader.label_map_path = "/gpfs/projects/ml/data/satdetect/astronet_label_map_2.pbtxt"
        self.input_reader.shuffle = False
        self.input_reader.num_readers = 1

def main(unused_argv):
    config_file = FLAGS.config_file
    print(config_file)

    config_generator = AstroNetFasterRcnnResnet101Generator(config_file)

    config_generator.batch_size(4)
    config_generator.initial_learning_rate(0.003)
    config_generator.config_file_output()

    print('main: Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str, 
                        default='/object_detection/protos_test/faster_rcnn_resnet101_astronet_test.config', 
                        help='astronet config file name')
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)



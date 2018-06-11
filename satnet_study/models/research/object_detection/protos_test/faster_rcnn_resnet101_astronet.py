#! /usr/bin/env python

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


# Main procedure:  Reads the entire address book from a file,
#   adds one person based on user input, then writes it back out to the same
#   file.
if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "FASTER_RCNN_RESNET101_ASTRONET_CONFIG_FILE")
  sys.exit(-1)

faster_rcnn_astronet = faster_rcnn_resnet101_astronet_pb2.FasterRCNNResnet101AstroNet()
detection_model = model_pb2.DetectionModel()
faster_rcnn = faster_rcnn_pb2.FasterRcnn()
train_config = train_pb2.TrainConfig()
eval_config = eval_pb2.EvalConfig()
train_input_reader = input_reader_pb2.InputReader()
eval_input_reader = input_reader_pb2.InputReader()
train_tf_record_input_reader = input_reader_pb2.TFRecordInputReader()
eval_tf_record_input_reader = input_reader_pb2.TFRecordInputReader()
optimizer = optimizer_pb2.Optimizer()
image_resizer = image_resizer_pb2.ImageResizer()
fixed_shape_resizer = image_resizer_pb2.FixedShapeResizer()
feature_extractor = faster_rcnn_pb2.FasterRcnnFeatureExtractor()
anchor_generator = anchor_generator_pb2.AnchorGenerator()
grid_anchor_generator = grid_anchor_generator_pb2.GridAnchorGenerator()
hyperparams_1st_stage = hyperparams_pb2.Hyperparams()
hyperparams_2nd_stage = hyperparams_pb2.Hyperparams()
regularizer_1st_stage = hyperparams_pb2.Regularizer()
regularizer_2nd_stage = hyperparams_pb2.Regularizer()
l2_regularizer_1st_stage = hyperparams_pb2.L2Regularizer()
l2_regularizer_2nd_stage = hyperparams_pb2.L2Regularizer()
initializer_1st_stage = hyperparams_pb2.Initializer()
initializer_2nd_stage = hyperparams_pb2.Initializer()
truncated_normal_initializer = hyperparams_pb2.TruncatedNormalInitializer()
variance_scaling_initializer = hyperparams_pb2.VarianceScalingInitializer()
box_predictor = box_predictor_pb2.BoxPredictor()
mask_rcnn_box_predictor = box_predictor_pb2.MaskRCNNBoxPredictor()
post_processing = post_processing_pb2.PostProcessing()
batch_non_max_suppression = post_processing_pb2.BatchNonMaxSuppression()

fixed_shape_resizer.height = 512
fixed_shape_resizer.width = 512

image_resizer.fixed_shape_resizer.CopyFrom(fixed_shape_resizer)

feature_extractor.type = 'faster_rcnn_resnet101'
feature_extractor.first_stage_features_stride = 16

grid_anchor_generator.height_stride = 16
grid_anchor_generator.width_stride = 16
grid_anchor_generator.scales[:] = [0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 1.0, 2.0]
grid_anchor_generator.aspect_ratios.append(1.0) 

anchor_generator.grid_anchor_generator.CopyFrom(grid_anchor_generator)

hyperparams_1st_stage.op = hyperparams_pb2.Hyperparams.CONV
hyperparams_2nd_stage.op = hyperparams_pb2.Hyperparams.FC
truncated_normal_initializer.stddev = 0.01
variance_scaling_initializer.factor = 1.0
variance_scaling_initializer.uniform = True
variance_scaling_initializer.mode = hyperparams_pb2.VarianceScalingInitializer.FAN_AVG
initializer_1st_stage.truncated_normal_initializer.CopyFrom(truncated_normal_initializer)
initializer_2nd_stage.variance_scaling_initializer.CopyFrom(variance_scaling_initializer)
hyperparams_1st_stage.initializer.CopyFrom(initializer_1st_stage)
hyperparams_2nd_stage.initializer.CopyFrom(initializer_2nd_stage)
l2_regularizer_1st_stage.weight = 0.0
l2_regularizer_2nd_stage.weight = 0.0
regularizer_1st_stage.l2_regularizer.CopyFrom(l2_regularizer_1st_stage)
regularizer_2nd_stage.l2_regularizer.CopyFrom(l2_regularizer_2nd_stage)
hyperparams_1st_stage.regularizer.CopyFrom(regularizer_1st_stage)
hyperparams_2nd_stage.regularizer.CopyFrom(regularizer_2nd_stage)

batch_non_max_suppression.score_threshold = 0.0
batch_non_max_suppression.iou_threshold = 0.6
batch_non_max_suppression.max_detections_per_class = 100
batch_non_max_suppression.max_total_detections = 100

post_processing.batch_non_max_suppression.CopyFrom(batch_non_max_suppression)
post_processing.score_converter = post_processing_pb2.PostProcessing.SOFTMAX

faster_rcnn.num_classes = 1
faster_rcnn.image_resizer.CopyFrom(image_resizer)
faster_rcnn.feature_extractor.CopyFrom(feature_extractor)
faster_rcnn.first_stage_anchor_generator.CopyFrom(anchor_generator)
faster_rcnn.first_stage_box_predictor_conv_hyperparams.CopyFrom(hyperparams_1st_stage)

faster_rcnn.first_stage_nms_score_threshold = 0.0
faster_rcnn.first_stage_nms_iou_threshold = 0.7
faster_rcnn.first_stage_max_proposals = 300
faster_rcnn.first_stage_localization_loss_weight = 2.0
faster_rcnn.first_stage_objectness_loss_weight = 1.0
faster_rcnn.initial_crop_size = 14
faster_rcnn.maxpool_kernel_size = 2
faster_rcnn.maxpool_stride = 2

mask_rcnn_box_predictor.fc_hyperparams.CopyFrom(hyperparams_2nd_stage)
mask_rcnn_box_predictor.use_dropout = False
mask_rcnn_box_predictor.dropout_keep_probability = 1.0

box_predictor.mask_rcnn_box_predictor.CopyFrom(mask_rcnn_box_predictor)

faster_rcnn.second_stage_box_predictor.CopyFrom(box_predictor)
faster_rcnn.second_stage_post_processing.CopyFrom(post_processing)
faster_rcnn.second_stage_localization_loss_weight = 2.0
faster_rcnn.second_stage_classification_loss_weight = 1.0

train_tf_record_input_reader.input_path.append("/gpfs/projects/ml/data/satdetect/astronet_train_3.tfrecords")
eval_tf_record_input_reader.input_path.append("/gpfs/projects/ml/data/satdetect/astronet_valid_3.tfrecords")

train_input_reader.tf_record_input_reader.CopyFrom(train_tf_record_input_reader)
eval_input_reader.tf_record_input_reader.CopyFrom(eval_tf_record_input_reader)

train_input_reader.label_map_path = "/gpfs/projects/ml/data/satdetect/astronet_label_map_2.pbtxt"
eval_input_reader.label_map_path = "/gpfs/projects/ml/data/satdetect/astronet_label_map_2.pbtxt"
eval_input_reader.shuffle = False
eval_input_reader.num_readers = 1

exponential_decay_learning_rate = optimizer_pb2.ExponentialDecayLearningRate()
exponential_decay_learning_rate.initial_learning_rate = 0.002

learning_rate = optimizer_pb2.LearningRate()
learning_rate.exponential_decay_learning_rate.CopyFrom(exponential_decay_learning_rate)

adam_optimizer = optimizer_pb2.AdamOptimizer()
adam_optimizer.learning_rate.CopyFrom(learning_rate)

optimizer.adam_optimizer.CopyFrom(adam_optimizer)

train_config.batch_size = 8
train_config.optimizer.CopyFrom(optimizer)
train_config.batch_queue_capacity = 8 
train_config.num_batch_queue_threads = 8
train_config.fine_tune_checkpoint = "/gpfs/projects/ml/astro_net/pretrained_resnet101/model.ckpt"
train_config.from_detection_checkpoint = True
train_config.gradient_clipping_by_norm = 10.0
train_config.num_steps = 0 # indefinitely
train_config.max_number_of_boxes = 100 

eval_config.num_visualizations = 20
eval_config.max_evals = 0 # indefinitely
eval_config.num_examples = 5000

# Add a model.
detection_model.faster_rcnn.CopyFrom(faster_rcnn)
faster_rcnn_astronet.model.CopyFrom(detection_model)
faster_rcnn_astronet.train_config.CopyFrom(train_config)
faster_rcnn_astronet.eval_config.CopyFrom(eval_config)
faster_rcnn_astronet.train_config.CopyFrom(train_config)
faster_rcnn_astronet.eval_config.CopyFrom(eval_config)
faster_rcnn_astronet.train_input_reader.CopyFrom(train_input_reader)
faster_rcnn_astronet.eval_input_reader.CopyFrom(eval_input_reader)

print('faster_rcnn_astronet:')
print(faster_rcnn_astronet)

# Write the model back to disk.
with open(sys.argv[1], "w") as f:
  f.write(text_format.MessageToString(faster_rcnn_astronet))

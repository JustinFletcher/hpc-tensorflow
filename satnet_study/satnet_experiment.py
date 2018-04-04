
import os
import sys
import csv
import glob
import argparse
import tensorflow as tf


def tensorflow_experiment():

    # Clear existing directory.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # The path to the training script.
    # train_script='/gpfs/home/fletch/hpc-tensorflow/resnet_cifar_study/models/official/resnet/cifar10_main.py'

    # The script expects a model_dir, use log_dir.
    # FLAGS['train_dir'] = FLAGS.log_dir
    vars(FLAGS)['train_dir'] = FLAGS.log_dir

    # # These flags are acceptable to the training script provided by TF.
    # script_flags = ['h',
    #                 'data_dir',
    #                 'model_dir',
    #                 'train_epochs',
    #                 'epochs_per_eval',
    #                 'batch_size',
    #                 'multi_gpu',
    #                 'hooks',
    #                 'num_parallel_calls',
    #                 'inter_op_parallelism_threads',
    #                 'intra_op_parallelism_threads',
    #                 'use_synthetic_data',
    #                 'max_train_steps',
    #                 'data_format',
    #                 'version',
    #                 'resnet_size']

    # # These are the summary tags to store.
    # summaries_to_store = ['global_step/sec',
    #                       'loss']

    # # Initialize an empty sting.
    # flags_string = ""

    # # Iterate over the input flags... If in 2.7, use vars(FLAGS).iteritems()
    # for key, value in vars(FLAGS).items():

    #     # If the input flag is acceptable for this script...
    #     if key in script_flags:

    #         # ...append it to the string.
    #         flags_string += " --%s=%s" % (key, value)

    # Initialize an empty sting.
    flags_string = ""

    # Iterate over the input flags... If in 2.7, use vars(FLAGS).iteritems()
    for key, value in vars(FLAGS).items():

        # ...append it to the string.
        flags_string += " --%s=%s" % (key, value)

    for c in FLAGS.num_cycles:

        # Run the training script with the constructed flag string, blocking.
        print("Calling: python %s %s" % (FLAGS.train_script, flags_string))

        # Call train in barckground.
        os.system("python %s %s" % (FLAGS.train_script, flags_string))

        # Call eval.
        os.system("python %s %s" % (FLAGS.eval_script, flags_string))


    # Get a list of events filenames in the model_dir.
    events_file_list = glob.glob(FLAGS.log_dir + 'events.out.tfevents.*')

    print("Event file list")
    print(events_file_list)

    # Write the data we saved to a csv file.
    with open(FLAGS.log_dir + FLAGS.log_filename, 'w') as csvfile:


def main(_):

    return(tensorflow_experiment())


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.
    parser.add_argument('--train_script', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_study/models/research/object_detection/train.py',
                        help='The core training script.')

    parser.add_argument('--eval_script', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_study/models/research/object_detection/eval.py',
                        help='The core training script.')

    parser.add_argument('--log_dir', type=str,
                        default='/gpfs/projects/ml/log/satnet_study_local/',
                        help='Model checkpoint and event directory.')

    parser.add_argument('--pipeline_config_path', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_study/faster_rcnn_resnet101_astronet.config',
                        help='Path to pipeling config.')

    parser.add_argument('--log_filename', type=str,
                        default='defualt.csv',
                        help='Merged output filename.')

    parser.add_argument('--num_cycles', type=int,
                        default=10,
                        help='Number of times to repeat train and eval cycle.')


    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

#!/usr/bin/python
# Example PBS cluster job submission in Python

import argparse
import tensorflow as tf
import os
import sys


dir_path1 = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path1)
print('satnet_cluster_experimet_launcher.py dir_path1:', dir_path1)

os.chdir('models/research')

cwd = os.getcwd()
print('cwd:', cwd)

dir_path2 = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path2)
print('satnet_cluster_experimet_launcher.py dir_path2:', dir_path2)

from satnet_cluster_experiment import SatnetClusterExperiment

os.chdir(dir_path1)

def main(FLAGS):

    # Clear and remake the log directory.
    # if tf.gfile.Exists(FLAGS.log_dir):

    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)

    # tf.gfile.MakeDirs(FLAGS.log_dir)

    # Instantiate an experiment.
    exp = SatnetClusterExperiment()

    # Set the number of reps for each config.
    # exp.set_rep_count(2)
    # Set independent parameters.
    # exp.add_design('batch_size', [8,16,32,64,128,256])
    # exp.add_design('num_batch_queue_threads', [8,16,32])
    # exp.add_design('initial_learning_rate', [0.001,0.002,0.004,0.008])
    # exp.add_design('num_steps', [1000,2000,4000,8000])
    # exp.add_design('batch_queue_capacity', [8,16,32])
    # exp.add_design('max_number_of_boxes', [25,50,75,100])
    exp.add_design('batch_size', [8])
    exp.add_design('num_batch_queue_threads', [8])
    exp.add_design('initial_learning_rate', [0.001])
    exp.add_design('num_steps', [1000,2000,4000,8000])
    exp.add_design('batch_queue_capacity', [8])
    exp.add_design('max_number_of_boxes', [25])
    # exp.add_design('test_interval', [100])
    # exp.add_design('pause_time', [10])
    # exp.add_design('model_name', ['faster_rcnn_resnet101_astronet'])

    # Launch the experiment.
    exp.launch_experiment(exp_filename=FLAGS.experiment_py_file,
                          log_dir=FLAGS.log_dir,
                          account_str='MHPCC96650DE1',
                          queue_str='standard',
                          module_str='anaconda2/5.0.1 gcc/5.3.0 cudnn/6.0',
                          config_file_dir = FLAGS.config_file_dir,
                          manager='pbs',
                          shuffle_job_order=True)

    # Manually note response varaibles.
    response_labels = ['step_num',
                       'train_loss',
                       'train_error',
                       'val_loss',
                       'val_error',
                       'mean_running_time',
                       'queue_size',
                       'mean_enqueue_rate',
                       'mean_dequeue_rate']

    # Wait for the output to return.
    exp.join_job_output(FLAGS.log_dir,
                        FLAGS.log_filename,
                        FLAGS.max_runtime,
                        response_labels)

    print("All jobs complete. Exiting.")


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str,
                        default='/gpfs/projects/ml/log/satnet_cluster_study_gm_1/trial-1/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='satnet_cluster_experiment.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=3600000,
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--experiment_py_file', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_study/satnet_cluster_experiment_example.py',
                        help='The satnet cluster experiment example.')

    parser.add_argument('--config_file_dir', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_study/satnet_cluster_configs/',
                        help='The config file directory.')

    # parser.add_argument('--config_file_dir', type=str,
    #                     default='~/tensorflow/models/research/astro_net/hokulea/satnet_cluster_configs/',
    #                     help='The config file directory.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

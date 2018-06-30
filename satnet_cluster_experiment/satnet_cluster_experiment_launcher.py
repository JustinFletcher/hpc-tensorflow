#!/usr/bin/python
# Example PBS cluster job submission in Python

import argparse
import tensorflow as tf

from cluster_experiment import ClusterExperiment


def main(FLAGS):

    # Clear and remake the log directory.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Instantiate an experiment.
    exp = ClusterExperiment()

    # Set the number of reps for each config.
    exp.set_rep_count(1)
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
                          module_str='anaconda3/5.0.1 tensorflow/1.8.0',
                          manager='pbs',
                          shuffle_job_order=True)

    # Manually note response variables.
    response_labels = ['global_step',
                       'total_loss',
                       'avg_precision']

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
                        default='/gpfs/projects/ml/log/satnet_cluster_experiment/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='satnet_cluster_experiment.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=3600000,
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--experiment_py_file', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_cluster_experiment/satnet_experiment.py',
                        help='The satnet cluster experiment.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

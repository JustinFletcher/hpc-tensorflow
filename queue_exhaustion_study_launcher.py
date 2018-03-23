#!/usr/bin/python
# Example PBS cluster job submission in Python

import argparse
import tensorflow as tf

# import pyhpc as hpc
# hpc.ClusterExperiment()

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
    exp.add_design('train_batch_size', [128, 256])
    # exp.add_design('batch_interval', [1, 2, 4, 8, 16, 32, 64, 128])
    # exp.add_design('train_enqueue_threads', [1, 2, 4, 8, 16, 32, 64, 128])
    exp.add_design('learning_rate', [0.0001])
    exp.add_design('max_steps', [10000])
    exp.add_design('test_interval', [100])
    exp.add_design('pause_time', [10])

    # Launch the experiment.
    exp.launch_experiment(exp_filename=FLAGS.experiment_py_file,
                          log_dir=FLAGS.log_dir,
                          account_str='MHPCC96650DE1',
                          queue_str='standard',
                          module_str='anaconda3/5.0.1 tensorflow',
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
                        default='../log/queue_exhaustion_study/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='queue_exhaustion_study.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=3600000,
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--experiment_py_file', type=str,
                        default='~/hpc-tensorflow/queue_exhaustion_study_experiment.py',
                        help='Number of seconds to run before giving up.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

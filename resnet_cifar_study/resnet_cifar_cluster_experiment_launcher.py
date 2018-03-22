#!/usr/bin/python
# Example PBS cluster job submission in Python

import os
import sys
import argparse
import tensorflow as tf

# import pyhpc as hpc
# hpc.ClusterExperiment()

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from cluster_experiment import ClusterExperiment


def main(FLAGS):

    # Clear and remake the log directory.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Instantiate an experiment.
    exp = ClusterExperiment()

    # Set the number of reps for each config.
    exp.set_rep_count(2)
    # Set independent parameters.
    exp.add_design('batch_size', [128, 256])
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
                       'global_step_per_sec',
                       'loss']

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
                        default='/gpfs/projects/ml/log/resnet_cifar_study/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='resnet_cifar_study.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=3600000,
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--experiment_py_file', type=str,
                        default='/gpfs/home/fletch/hpc-tensorflow/resnet_cifar_study/resnet_cifar_experiment.py',
                        help='Number of seconds to run before giving up.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

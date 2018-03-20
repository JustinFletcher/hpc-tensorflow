import os
import sys
import csv
import glob
import argparse
import tensorflow as tf


def tensorflow_experiment():

    # Clear existing directory.
    if tf.gfile.Exists(FLAGS.model_dir):

        tf.gfile.DeleteRecursively(FLAGS.model_dir)

    tf.gfile.MakeDirs(FLAGS.model_dir)

    # The path to the training script.
    # train_script='/gpfs/home/fletch/hpc-tensorflow/resnet_cifar_study/models/official/resnet/cifar10_main.py'

    # These flags are acceptable to the training script provided by TF.
    script_flags = ['h',
                    'data_dir',
                    'model_dir',
                    'train_epochs',
                    'epochs_per_eval',
                    'batch_size',
                    'multi_gpu',
                    'hooks',
                    'num_parallel_calls',
                    'inter_op_parallelism_threads',
                    'intra_op_parallelism_threads',
                    'use_synthetic_data',
                    'max_train_steps',
                    'data_format',
                    'version',
                    'resnet_size']

    # Initialize an empty sting.
    flags_string = ""

    # Iterate over the input flags... If in 2.7, use vars(FLAGS).iteritems()
    for key, value in vars(FLAGS).items():

        # If the input flag is acceptable for this script...
        if key in script_flags:

            # ...append it to the string.
            flags_string += " --%s=%s" % (key, value)

    # Run the training script with the constructed flag string, blocking.
    os.system("python %s %s" % (FLAGS.train_script, flags_string))

    # Get a list of events filenames in the model_dir.
    events_file_list = glob.glob(FLAGS.model_dir + 'events.out.tfevents.*')

    print("Event file list")
    print(events_file_list)

    # Iterate over the event files in the model_dir.
    for ef in events_file_list:

        print("ef")
        print(ef)

        # Write the data we saved to a csv file.
        with open(FLAGS.log_dir + FLAGS.log_filename, 'wb') as csvfile:

            # Open a writer and write the header.
            csvwriter = csv.writer(csvfile)

            # Initialize an empty list to store summaries.
            row = []

            # TODO: Specify path to specific events file.
            # TODO: Extract all events files in the model_dir.
            # TODO: Find the latest event file.
            # TODO: Iterate over the summaries in that file.
            for e in tf.train.summary_iterator(ef):

                print(e)
                for v in e.summary.value:

                    # TODO: Add Step.

                    print(v)

                    if v.tag == 'loss':
                        print(v.simple_value)
                        row.append(v.simple_value)

                    # TODO: Add running time.

                csvwriter.writerow(row)

        # # Iterate over the results vectors for each config.
        # for (step, tl, te, vl, ve, mrt, qs, mer, mdr) in zip(steps,
        #                                                      train_losses,
        #                                                      train_errors,
        #                                                      val_losses,
        #                                                      val_errors,
        #                                                      mean_running_times,
        #                                                      queue_sizes,
        #                                                      mean_enqueue_rates,
        #                                                      mean_dequeue_rates):

        #     # Write the data to a csv.
        #     csvwriter.writerow([step,
        #                         tl,
        #                         te,
        #                         vl,
        #                         ve,
        #                         mrt,
        #                         qs,
        #                         mer,
        #                         mdr])


def main(_):

    return(tensorflow_experiment())


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.
    parser.add_argument('--train_script', type=str,
                        default='/gpfs/home/fletch/hpc-tensorflow/resnet_cifar_study/models/official/resnet/cifar10_main.py',
                        help='The core training script that this script wraps.')

    parser.add_argument('--model_dir', type=str,
                        default='/gpfs/projects/ml/tfmodels/resnet_cifar_model/',
                        help='Model checkpoint and event directory.')

    # These flags specify the data used in the experiment.
    parser.add_argument('--train_epochs', type=int,
                        default=6,
                        help='Directory from which to pull data TFRecords.')

    parser.add_argument('--epochs_per_eval', type=int,
                        default=2,
                        help='Directory from which to pull data TFRecords.')

    # These flags specify the data used in the experiment.
    parser.add_argument('--data_dir', type=str,
                        default='/gpfs/projects/ml/data/cifar10_official',
                        help='Directory from which to pull data TFRecords.')

    parser.add_argument('--log_dir', type=str,
                        default='/gpfs/projects/ml/log/resnet_cifar_study/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='defualt.csv',
                        help='Merged output filename.')



    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

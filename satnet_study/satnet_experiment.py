
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

    # TODO: Identify any flags passed in that match .congif names
    # TODO: Locally copy the global config
    # TODO: Update the local config

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

    for c in range(FLAGS.num_cycles):

        print("\n\n\n\n######## Begin Cycle #########")

        print("We are about to try to call train.py")

        # Run the training script with the constructed flag string, blocking.
        print("\n\n\n\nCalling: python %s %s" % (FLAGS.eval_script,
                                                 flags_string))

        # Call eval.
        os.system("python %s %s &" % (FLAGS.eval_script,
                                      flags_string))

        print("Now, we're calling test.py.")

        # Run the training script with the constructed flag string, blocking.
        print("\n\n\n\nCalling: python %s %s" % (FLAGS.train_script,
                                                 flags_string))

        # TODO: Make background.
        # Call train in barckground.
        os.system("python %s %s" % (FLAGS.train_script,
                                    flags_string))

        print("\n\n\n\n######## End Cycle #########")

    summaries_to_store = ['TotalLoss']

    # Get a list of events filenames in the model_dir.
    events_file_list = glob.glob(FLAGS.log_dir + 'events.out.tfevents.*')

    print("Event file list")
    print(events_file_list)

    # Write the data we saved to a csv file.
    with open(FLAGS.log_dir + FLAGS.log_filename, 'w') as csvfile:

        # Open a writer and write the header.
        csvwriter = csv.writer(csvfile)

        # Initialize placeholders.
        row = []
        current_step = -1

        # Iterate over the event files in the model_dir.
        for ef in events_file_list:

            # print("--New event file")
            # print(ef)

            # Iterate over each summary file in the model dir.
            for e in tf.train.summary_iterator(ef):

                # print("====Start e from tf.train.summary_iterator(ef)======")
                # print(e.step)

                # Parse the step.
                step = e.step

                # Check if this iterator has yielded a new step...
                if step > current_step:

                    # ...if so, write out the prior row...
                    if len(row) == len(summaries_to_store) + 1:

                        csvwriter.writerow(row)

                    # ...then clear the row storage...
                    row = []

                    # ...and append and update the step.
                    row.append(step)
                    current_step = step

                # Iterate over each summary value.
                for v in e.summary.value:

                    # print("=======Start v from e.summary.value========")
                    # print(v.tag)

                    # Check if present summary is in the summary list.
                    if v.tag in summaries_to_store:

                        # If so, append them.
                        row.append(v.simple_value)

                    # print("=======End v from e.summary.value========")

                # print("====End e from tf.train.summary_iterator(ef)======")


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
                        help='The core evaluation script.')

    parser.add_argument('--log_dir', type=str,
                        default='/gpfs/projects/ml/log/satnet_study_gm_1/',
                        help='Model checkpoint and event directory.')

    parser.add_argument('--pipeline_config_path', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_study/faster_rcnn_resnet101_astronet.config',
                        help='Path to pipeling config.')

    parser.add_argument('--log_filename', type=str,
                        default='default.csv',
                        help='Merged output filename.')

    parser.add_argument('--num_cycles', type=int,
                        default=2,
                        help='Number of times to repeat train and eval cycle.')




    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

import os
import sys
import csv
import argparse
import importlib
import tensorflow as tf


def tensorflow_experiment():

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    train_script='/gpfs/home/fletch/hpc-tensorflow/resnet_cifar_study/models/official/resnet/cifar10_main.py'

    flags_string = ""
    for key, value in vars(FLAGS).items():
        flags_string += " --%s=%s" % (key, value)
    print(flags_string)

    print("I want to run: %s" % train_script)
    os.system("python %s %s" % (FLAGS.train_script, flags_string))
    print("I tried...")

    # Write the data we saved to a csv file, to be compiled by the launcher.
    with open(FLAGS.log_dir + '/' + FLAGS.log_filename, 'wb') as csvfile:

        # Open a writer and write the header.
        csvwriter = csv.writer(csvfile)

        row = []

        # TODO: Specify path to specific events file.
        for e in tf.train.summary_iterator(FLAGS.train_dir + 'events.out.tfevents.1521127912.hokulea02.mhpcc.hpc.mil'):

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
    parser.add_argument('--model_dir', type=str,
                        default='/gpfs/projects/ml/tfmodels/resnet_cifar_model/',
                        help='Model checkpoint and event directory.')

    # These flags specify the data used in the experiment.
    parser.add_argument('--data_dir', type=str,
                        default='/gpfs/projects/ml/data/cifar10_official',
                        help='Directory from which to pull data TFRecords.')



    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

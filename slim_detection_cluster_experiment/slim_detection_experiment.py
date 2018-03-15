import sys
import csv
import argparse
import tensorflow as tf


def tensorflow_experiment():

    # import FLAGS.slim_train_file

    ####################
    # Train a network. #
    ####################

    FLAGS.slim_train_file

    #########################
    # Evaluate the network. #
    #########################

    FLAGS.slim_eval_file

    ############################
    # Reads summaries to CSVs. #
    ############################

    
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)


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
    parser.add_argument('--slim_dir', type=str,
                        default='/gpfs/projects/ml/tfmodels/models/research/slim/',
                        help='Slim directory.')

    parser.add_argument('--slim_train_file', type=str,
                        default='/gpfs/projects/ml/tfmodels/models/research/slim/train_image_classifier.py',
                        help='Slim train file.')

    parser.add_argument('--slim_eval_file', type=str,
                        default='/gpfs/projects/ml/tfmodels/models/research/slim/eval_image_classifier.py',
                        help='Slim eval file.')

    parser.add_argument('--train_dir', type=str,
                        default='/gpfs/projects/ml/tfmodels/cifarnet-model/',
                        help='Model checkpoint and event directory.')

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--log_dir', type=str,
                        default='/gpfs/projects/ml/tmp/slim_test/',
                        help='Summaries log directory.')

    parser.add_argument('--pause_time', type=float,
                        default=0.0,
                        help='Number of seconds to pause before execution.')

    parser.add_argument('--log_filename', type=str,
                        default='deep_sa_generalization_experiment.csv',
                        help='Summaries log directory.')

    parser.add_argument('--keep_prob', type=float,
                        default=1.0,
                        help='Keep probability for output layer dropout.')

    parser.add_argument('--train_batch_size', type=int,
                        default=128,
                        help='Training set batch size.')

    parser.add_argument('--batch_interval', type=int,
                        default=1,
                        help='Interval between training batch refresh.')

    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=100,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    # These flags specify the data used in the experiment.
    parser.add_argument('--data_dir', type=str,
                        default='/gpfs/projects/ml/data/cifar10',
                        help='Directory from which to pull data TFRecords.')

    parser.add_argument('--train_file', type=str,
                        default='train.tfrecords',
                        help='Training dataset filename.')

    parser.add_argument('--validation_file', type=str,
                        default='validation.tfrecords',
                        help='Validation dataset filename.')

    parser.add_argument('--input_size', type=int,
                        default=28 * 28,
                        help='Dimensionality of the input space.')

    parser.add_argument('--label_size', type=int,
                        default=10,
                        help='Dimensinoality of the output space.')

    parser.add_argument('--val_batch_size', type=int,
                        default=10000,
                        help='Validation set batch size.')

    # These flags control the input pipeline threading.
    parser.add_argument('--val_enqueue_threads', type=int,
                        default=32,
                        help='Number of threads to enqueue val examples.')

    parser.add_argument('--train_enqueue_threads', type=int,
                        default=128,
                        help='Number of threads to enqueue train examples.')

    # These flags specify placekeeping variables.
    parser.add_argument('--rep_num', type=int,
                        default=0,
                        help='Flag identifying the repitition number.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

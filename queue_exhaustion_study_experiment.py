

import sys
import csv
import time
import argparse
import numpy as np
import tensorflow as tf

# Change baseline_model to a model you built with the same interfaces.
from baseline_model import *

# import tfmodelzoo as zoo
# model_builder = zoo.get_untrained_model("lenet5")
# model = model_builder(FLAGS.input_size,
#               FLAGS.label_size,
#               FLAGS.learning_rate,
#               FLAGS.train_enqueue_threads,
#               FLAGS.val_enqueue_threads,
#               FLAGS.data_dir,
#               FLAGS.train_file,
#               FLAGS.validation_file)


def tensorflow_experiment():

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Reset the default graph.
    tf.reset_default_graph()

    # Declare experimental measurement variables as lists.
    steps = []
    val_losses = []
    val_errors = []
    train_losses = []
    train_errors = []
    mean_running_times = []
    mean_enqueue_rates = []
    mean_dequeue_rates = []
    queue_sizes = []

    print("------------model_output-------------")
    # Instantiate a model.
    model = Model(FLAGS.input_size,
                  FLAGS.label_size,
                  FLAGS.learning_rate,
                  FLAGS.train_enqueue_threads,
                  FLAGS.val_enqueue_threads,
                  FLAGS.data_dir,
                  FLAGS.train_file,
                  FLAGS.validation_file)
    print("-------------------------------------")

    # Get input data.
    image_batch, label_batch = model.get_train_batch_ops(
        batch_size=FLAGS.train_batch_size)

    (val_image_batch, val_label_batch) = model.get_val_batch_ops(
        batch_size=FLAGS.val_batch_size)

    # Merge the summary.
    tf.summary.merge_all()

    # Get queue size Op.
    qr = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)[1].queue.size()

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=600.0)

    with sv.managed_session() as sess:

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir +
        #                                      '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        # Declare timekeeping vars.
        running_times = [0.0001]
        net_enqueue_rates = [0]
        running_time = 0

        # print("------------training_output-------------")

        # Print a line for debug.
        # print('step | train_loss | train_error | val_loss |' +
        #       ' val_error | t | total_time')

        # Load the validation set batch into memory.
        val_images, val_labels = sess.run([val_image_batch, val_label_batch])

        # Make a dict to load the val batch onto the placeholders.
        val_dict = {model.stimulus_placeholder: val_images,
                    model.target_placeholder: val_labels,
                    model.keep_prob: 1.0}

        time.sleep(FLAGS.pause_time)

        # Iterate until max steps.
        for i in range(FLAGS.max_steps):

            # Check for break.
            if sv.should_stop():
                break

            start_time = time.time()

            # Measure queue size.
            current_queue_size_net = sess.run(qr)

            # If it is a batch refresh interval, refresh the batch.
            if((i % FLAGS.batch_interval == 0) or (i == 0)):

                # Update the batch.
                train_images, train_labels = sess.run([image_batch,
                                                       label_batch])

            # Make a dict to load the batch onto the placeholders.
            train_dict = {model.stimulus_placeholder: train_images,
                          model.target_placeholder: train_labels,
                          model.keep_prob: FLAGS.keep_prob}

            # If we have reached a testing interval, test.
            if (i % FLAGS.test_interval == 0):

                # Compute error over the training set.
                train_error = sess.run(model.error, feed_dict=train_dict)

                # Compute loss over the training set.
                train_loss = sess.run(model.loss, feed_dict=train_dict)

                # Compute error over the validation set.
                val_error = sess.run(model.error, feed_dict=val_dict)

                # Compute loss over the validation set.
                val_loss = sess.run(model.loss, feed_dict=val_dict)

                # Store the data we wish to manually report.
                steps.append(i)
                train_losses.append(train_loss)
                train_errors.append(train_error)
                val_losses.append(val_loss)
                val_errors.append(val_error)

                mean_running_time = np.mean(running_times)
                mean_running_times.append(mean_running_time)

                mean_net_enqueue_rate = np.mean(net_enqueue_rates)

                mean_dequeue_rate = FLAGS.train_batch_size / (mean_running_time * FLAGS.batch_interval)

                mean_enqueue_rates.append(mean_net_enqueue_rate + mean_dequeue_rate)
                mean_dequeue_rates.append(mean_dequeue_rate)

                current_queue_size = sess.run(qr)
                queue_sizes.append(current_queue_size)

                # Reset running times measurment
                running_times = []
                net_enqueue_rates = []

                # Print relevant values.
                # print('%d | %.6f | %.2f | %.6f | %.2f | %.6f | %.2f'
                #       % (i,
                #          train_loss,
                #          train_error,
                #          val_loss,
                #          val_error,
                #          np.mean(running_times),
                #          np.sum(running_times))) 

            # Optimize the model.
            sess.run(model.optimize, feed_dict=train_dict)

            # train_writer.add_summary(summary, i)

            # Update timekeeping variables.
            running_time = time.time() - start_time
            running_times.append(running_time)

            # Measure the queue now.
            final_queue_size_net = sess.run(qr)

            # Compute and append the dequeue rate.
            net_enqueue_rate = (final_queue_size_net - current_queue_size_net) / running_time
            net_enqueue_rates.append(net_enqueue_rate)

        print("----------------------------------------")
        # Close the summary writers.
        # test_writer.close()
        # train_writer.close()
        sv.stop()
        sess.close()

    # Write the data we saved to a csv file, to be compiled by the launcher.
    with open(FLAGS.log_dir + '/' + FLAGS.log_filename, 'wb') as csvfile:

        # Open a writer and write the header.
        csvwriter = csv.writer(csvfile)

        # Iterate over the results vectors for each config.
        for (step, tl, te, vl, ve, mrt, qs, mer, mdr) in zip(steps,
                                                             train_losses,
                                                             train_errors,
                                                             val_losses,
                                                             val_errors,
                                                             mean_running_times,
                                                             queue_sizes,
                                                             mean_enqueue_rates,
                                                             mean_dequeue_rates):

            # Write the data to a csv.
            csvwriter.writerow([step,
                                tl,
                                te,
                                vl,
                                ve,
                                mrt,
                                qs,
                                mer,
                                mdr])

    return()


def main(_):

    return(tensorflow_experiment())


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--log_dir', type=str,
                        default='../log/tensorflow_experiment/templog',
                        help='Summaries log directory.')

    parser.add_argument('--pause_time', type=str,
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
                        default='../data/mnist',
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

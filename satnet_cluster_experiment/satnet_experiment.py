
import os
import sys
sys.path.append("../satnet_study/models/research")
import csv
import glob
import argparse
import tensorflow as tf

from object_detection.protos_test import faster_rcnn_resnet101_astronet_generator

def tensorflow_experiment():

    # Clear existing directory.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)
    
    FLAGS.eval_dir = FLAGS.log_dir + 'eval/' 
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # create the config file:
    pipeline_config_path = FLAGS.log_dir + 'faster_rcnn_resnet101_astronet.config'

    config_generator = faster_rcnn_resnet101_astronet_generator.AstroNetFasterRcnnResnet101Generator(pipeline_config_path)

    # print('batch_size:',FLAGS.batch_size)
    # print('initial_learning_rate:',FLAGS.initial_learning_rate)
    # print('num_batch_queue_threads:',FLAGS.num_batch_queue_threads)
    # print('batch_queue_capacity:',FLAGS.batch_queue_capacity)
    # print('num_steps:',FLAGS.num_steps)
    # print('max_number_of_boxes:',FLAGS.max_number_of_boxes)

    config_generator.batch_size(FLAGS.batch_size)
    config_generator.initial_learning_rate(FLAGS.initial_learning_rate)
    config_generator.num_batch_queue_threads(FLAGS.num_batch_queue_threads)
    config_generator.batch_queue_capacity(FLAGS.batch_queue_capacity)
    config_generator.num_steps(FLAGS.num_steps)
    config_generator.max_number_of_boxes(FLAGS.max_number_of_boxes)
    config_generator.config_file_output()

    # The script expects a model_dir, use log_dir.
    vars(FLAGS)['train_dir'] = FLAGS.log_dir
    vars(FLAGS)['eval_dir'] = FLAGS.eval_dir
    vars(FLAGS)['checkpoint_dir'] = FLAGS.log_dir
    vars(FLAGS)['pipeline_config_path'] = pipeline_config_path

    # Initialize an empty sting.
    train_flags_string = "--logtostderr "
    eval_flags_string  = "--logtostderr "

    train_flags_list = ['pipeline_config_path', 'train_dir']
    eval_flags_list  = ['pipeline_config_path', 'eval_dir', 'checkpoint_dir',]

    # Iterate over the input flags... If in 2.7, use vars(FLAGS).iteritems()
    # for key, value in vars(FLAGS).iteritems():

    for key, value in vars(FLAGS).items():

        if key in train_flags_list:
            train_flags_string += " --%s=%s" % (key, value)

        if key in eval_flags_list:
            eval_flags_string += " --%s=%s" % (key, value)

    for c in range(FLAGS.num_cycles):

        print("\n\n\n\n######## Begin Cycle #########")

        print("Call train.py.")

        # # Run the training script with the constructed flag string, blocking.
        print("\n\n\n\nCalling: python %s %s" % (FLAGS.train_script,
                                                 train_flags_string))

        # # TODO: Make background.
        # # Call train in background.
        # os.system("python %s %s &" % (FLAGS.train_script,
        #                             train_flags_string))
        # # Call train in foreground.
        os.system("python %s %s" % (FLAGS.train_script,
                                    train_flags_string))

        print("Call eval.py")

        # Run the evaluation script with the constructed flag string, blocking.
        print("\n\n\n\nCalling: python %s %s" % (FLAGS.eval_script,
                                                 eval_flags_string))

        # Call eval in background:.
        # os.system("python %s %s &" % (FLAGS.eval_script,
        #                               eval_flags_string))
        os.system("python %s %s" % (FLAGS.eval_script,
                                      eval_flags_string))

        print("\n\n\n\n######## End Cycle #########")

    training_summaries_to_store = ['TotalLoss']
    eval_summaries_to_store     = ['PascalBoxes_PerformanceByCategory/AP@0.5IOU/Satellite']
    # eval_summaries_to_store     = ['PascalBoxes_PerformanceByCategory/AP@0.5IOU/Satellite','PascalBoxes_Precision/mAP@0.5IOU']

    # Get a list of events filenames in the model_dir.
    events_file_list = glob.glob(FLAGS.log_dir + 'events.out.tfevents.*') + glob.glob(FLAGS.eval_dir + 'events.out.tfevents.*')

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

            isEval = False
            nPad = 1
            if 'eval' not in ef:
                summaries_to_store = training_summaries_to_store
                isEval = False
                nPad = 1
            else:
                summaries_to_store = eval_summaries_to_store
                isEval = True
                nPad = 2

            # Iterate over each summary file in the model dir.
            for e in tf.train.summary_iterator(ef):

                # print("====Start e from tf.train.summary_iterator(ef)======")
                # print(e.step)

                # Parse the step.
                step = e.step

                # print('step:',step)
                # print('current_step:', current_step)

                # Check if this iterator has yielded a new step...
                if step > current_step:

                    # ...if so, write out the prior row...
                    if len(row) == len(summaries_to_store) + nPad:

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
                        # print(v.tag,v.simple_value)
                        if isEval == True:
                            # for column alignment:
                            row.append(' ')
                        row.append(v.simple_value)

                    # print("=======End v from e.summary.value========")

                # print("====End e from tf.train.summary_iterator(ef)======")

            # clean up if row is not empty:
            if len(row) > 1:
                csvwriter.writerow(row)


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
                        default='/gpfs/projects/ml/log/satnet_cluster_experiment/templog/',
                        help='Model checkpoint and event directory.')

    parser.add_argument('--eval_dir', type=str,
                        default='/gpfs/projects/ml/log/satnet_cluster_experiment/templog/eval',
                        help='Eval directory.')

    parser.add_argument('--pipeline_config_path', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/satnet_cluster_experiment/templog/faster_rcnn_resnet101_astronet.config',
                        help='Path to pipeling config.')

    parser.add_argument('--log_filename', type=str,
                        default='templog.csv',
                        help='Log output filename.')

    parser.add_argument('--num_cycles', type=int,
                        default=1,
                        help='Number of times to repeat train and eval cycle.')

    parser.add_argument('--rep_num', type=int,
                        default=0,
                        help='Number of times to repeat train and eval cycle.')

    parser.add_argument('--batch_size', type=int,
                        default=8,
                        help='Number of images to process in a batch.')

    parser.add_argument('--num_batch_queue_threads', type=int,
                        default=8,
                        help='Number of enqueuing threads for batch reading.')

    parser.add_argument('--initial_learning_rate', type=float,
                        default=0.001,
                        help='The initial learning rate for the optimizer.')

    parser.add_argument('--num_steps', type=int,
                        default=500,
                        help='The maximum number of global steps to run during training.')

    parser.add_argument('--batch_queue_capacity', type=int,
                        default=8,
                        help='The queue capacity.')

    parser.add_argument('--max_number_of_boxes', type=int,
                        default=25,
                        help='The maximum number of object detections per image.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

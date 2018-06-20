
#!/usr/bin/python
# Example PBS cluster job submission in Python

import os
import sys
import csv
import time
import random
import argparse
import itertools
import subprocess

# path = os.path.abspath(__file__)
# dir_path = os.path.dirname(path)
# sys.path.append(dir_path)

# dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir('../..')

dir_path1 = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path1)

cwd = os.getcwd()
print('cwd:', cwd)

print('satnet_cluster_experimet.py dir_path1:', dir_path1)

# os.chdir('models/research')

cwd = os.getcwd()
print('cwd:', cwd)

dir_path2 = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path2)

print('satnet_cluster_experimet.py dir_path2:', dir_path2)

from object_detection.protos_test import faster_rcnn_resnet101_astronet_generator

os.chdir(dir_path1)

class SatnetClusterExperiment(object):

    def __init__(self):

        # self.experimental_configs = []
        self.independent_designs = []
        # self.coupled_designs = []
        self.parameter_labels = []
        self._input_output_maps = []
        self._job_ids = []

    def add_design(self, flag, value_list):

        self.independent_designs.append((flag, value_list))

    # def add_coupled_design(self, coupled_design):

    #     self.coupled_designs.append(coupled_design)

    # def set_rep_count(self, num_reps):

    #     self.add_design('rep_num', range(num_reps))

    def get_configs(self,config_file_dir):

        # Translate the design structure into flag strings.
        # exp_flag_strings = [['--' + f + '=' + str(v) for v in r]
        #                     for (f, r) in self.independent_designs]

        # Translate the design structure into a faster_rcnn_resnet101_astronet_generator command.
        exp_command_strings = [[f + '(' + str(v) + ')' for v in r]
                            for (f, r) in self.independent_designs]


        # print('exp_command_strings:')
        # print(exp_command_strings)

        # sys.exit()

        # Produce the Cartesian set of configurations.
        # indep_experimental_configs = itertools.product(*exp_command_strings)
        indep_experimental_configs = list(itertools.product(*exp_command_strings))

        nConfigs = len(indep_experimental_configs)

        # print('nConfigs:', nConfigs)

        # print('indep_experimental_configs[0]:')
        # print(indep_experimental_configs[0])

        # print('indep_experimental_configs[1]:')
        # print(indep_experimental_configs[1])

        # print('indep_experimental_configs[nConfigs-1]:')
        # print(indep_experimental_configs[nConfigs-1])

        # for each configuration, generate a config file and save the path in a list
        config_file_pathname_args_list = []

        config_file_basename = 'faster_rcnn_resnet101_astronet_test.config'
        # config_file_dir = FLAGS.config_file_dir
        # print('config_file_dir:', config_file_dir)
        config_generator_str = 'config_generator.'

        for i,cmd_tuple in enumerate(indep_experimental_configs):
            config_name = 'config' + '-' + str(i)
            config_file = config_file_dir + config_file_basename.replace('config', config_name)
            config_file = os.path.expanduser(config_file)
            # print('config_file:', config_file)
            # sys.exit()
            config_generator = faster_rcnn_resnet101_astronet_generator.AstroNetFasterRcnnResnet101Generator(config_file)
            for t in cmd_tuple:
                cmd = config_generator_str + t
                # print('cmd:', cmd)
                exec(cmd)
            config_generator.config_file_output()
            config_cmd_args = '--pipeline_config_path' + ' ' + config_file
            config_file_pathname_args_list.append(config_cmd_args)

        # print('config_file_pathname_args_list:')
        # print(config_file_pathname_args_list)

        # print('config_file_pathname_args_list[0]:')
        # print(config_file_pathname_args_list[0])

        # print('config_file_pathname_args_list[1]:')
        # print(config_file_pathname_args_list[1])

        # print('config_file_pathname_args_list[nConfigs-1]:')
        # print(config_file_pathname_args_list[nConfigs-1])

        # sys.exit()

        # Initialize a list to store coupled configurations.
        # coupled_configs = []

        # # Scope this variable higher due to write-out coupling.
        # coupled_flag_strs = []

        # for coupled_design in self.coupled_designs:

        #     for d in coupled_design:

        #         coupled_flag_strs = [['--' + f + '=' + str(v) for v in r]
        #                              for (f, r) in d]

        #         coupled_configs += list(itertools.product(*coupled_flag_strs))

        # Initialize empty experimental configs list...
        # experimental_configs = []

        # ...and if there are coupled configs...
        # if coupled_configs:

        #     # ...iterate over each independent config...
        #     for e in indep_experimental_configs:

        #             # ...then for each coupled config...
        #             for c in coupled_configs:

        #                 # ...join the coupled config to the independent one.
        #                 experimental_configs.append(e + tuple(c))

        # # Otherwise, ....
        # else:

        #     # ...just pass the independent experiments through.
        #     experimental_configs = indep_experimental_configs

        # return(experimental_configs)

        return(config_file_pathname_args_list)

    def get_parameter_labels(self):

        parameter_labels = []

        for (f, _) in self.independent_designs:

            parameter_labels.append(f)

        # for coupled_design in self.coupled_designs:

        #     coupled_flag_strs = []

        #     for d in coupled_design:

        #         coupled_flag_strs = [f for (f, _) in d]

        #     parameter_labels += coupled_flag_strs

        return(parameter_labels)

    def launch_experiment(self,
                          exp_filename,
                          log_dir,
                          account_str,
                          queue_str,
                          module_str,
                          config_file_dir,
                          manager='pbs',
                          shuffle_job_order=True):

        experimental_configs = self.get_configs(config_file_dir)

        if shuffle_job_order:

            # Shuffle the submission order of configs to avoid asymetries.
            random.shuffle(experimental_configs)

        # Iterate over each experimental configuration, launching jobs.
        # for i, experimental_config in enumerate(experimental_configs):
        for experimental_config in experimental_configs:

            if manager == 'pbs':

                # Some prints for easy debug.
                print("-----------experimental_config---------")
                print(experimental_config)
                print("---------------------------------------")

                # find the config file index:
                # print('experimental_config:',experimental_config)

                strSearch = 'config-'
                nStrSearch = len(strSearch)
                # print('strSearch:',strSearch)
                # print('nStrSearch:',nStrSearch)

                strConfigIdx = experimental_config.find(strSearch)
                nConfig = len(experimental_config)
                # print('strConfigIdx:',strConfigIdx)
                # print('nConfig:',nConfig)

                strtIdx = strConfigIdx + nStrSearch
                endIdx = nConfig  
                # print('strtIdx:',strtIdx)
                # print('endIdx:',endIdx)

                strI = experimental_config[strtIdx:endIdx]
                # print('strI:',strI)

                i = int(strI)
                # print('i:',i)

                # Use subproces to command qsub to submit a job.
                p = subprocess.Popen('qsub',
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     shell=True)

                # Customize your options here.
                job_name = "satnet_ex_%d" % i
                # walltime = "72:00:00"
                walltime = "2:00:00"
                select = "1:ncpus=20:mpiprocs=20"
                command = "python " + exp_filename

                # Iterate over flag strings, building the command.
                # for flag in experimental_config:

                #     command += ' ' + flag

                command += ' ' + experimental_config

                # Add a final flag modifying the log filename to be unique.
                log_filename = 'templog' + str(i) + '.csv'

                temp_log_dir = log_dir + 'templog' + str(i) + '/'

                command += ' --log_dir=' + temp_log_dir

                # Add the logfile to the command.
                command += ' --log_filename=' + log_filename

                eval_dir = temp_log_dir + 'eval/'

                command += ' --eval_dir=' + eval_dir

                if not os.path.exists(temp_log_dir):
                    os.makedirs(temp_log_dir)

                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)

                # Build IO maps.
                input_output_map = (experimental_config,
                                    temp_log_dir,
                                    log_filename)

                self._input_output_maps.append(input_output_map)

                # Build the job string.
                job_string = '#!/bin/bash' + '\n'
                job_string += '#PBS -N ' + job_name + '\n'
                job_string += '#PBS -l walltime=' + walltime + '\n'
                job_string += '#PBS -l select=' + select + '\n'
                job_string += '#PBS -o ~/log/output/' + job_name + '.out \n'
                job_string += '#PBS -e ~/log/error/' + job_name + '.err \n'

                job_string += '#PBS -A ' + account_str + '\n'
                job_string += '#PBS -q ' + queue_str + '\n'
                job_string += 'module load ' + module_str + '\n'

                job_string += 'cd $PBS_O_WORKDIR ' + '\n'

                job_string += command

                # Remove below.
                # job_string = """#!/bin/bash
                # #PBS -N %s
                # #PBS -l walltime=%s
                # #PBS -l select=%s
                # #PBS -o ~/log/output/%s.out
                # #PBS -e ~/log/error/%s.err
                # #PBS -A MHPCC96650DE1
                # #PBS -q standard
                # module load anaconda3/5.0.1 gcc/5.3.0 cudnn/6.0
                # module unload anaconda2/5.0.1
                # cd $PBS_O_WORKDIR
                # %s""" % (job_name,
                #          walltime,
                #          select,
                #          job_name,
                #          job_name,
                #          command)

                # Print your job string.
                print(job_string)
                print("about to bytes")
                # job_string = bytes(job_string, 'utf-8')
                job_string = bytes(job_string)
                # print(job_string)

                # Send job_string to qsub, returning a job ID in bytes.
                job_id = p.communicate(job_string)[0]

                # print('job_id:')
                # print(job_id)
                # Parse the bytes to an ID string.
                job_id = job_id.decode("utf-8")
                # print('job_id:')
                # print(job_id)
                job_id = job_id.replace('\n','')

                # Append the job ID.
                self._job_ids.append(job_id)

                print(self._job_ids)
                time.sleep(0.1)

            else:

                print("Unknown manager supplied to launch_experiment().")
                exit()

# qsub -I -A MHPCC96650DE1 -q standard -l select=1:ncpus=20:mpiprocs=20 -l walltime=1:00:00

    def join_job_output(self,
                        log_dir,
                        log_filename,
                        max_runtime,
                        response_labels):

        jobs_complete = False
        timeout = False
        elapsed_time = 0

        # Loop until timeout or all jobs complete.
        while not(jobs_complete) and not(timeout):

            print("-----------------")

            print('Time elapsed: ' + str(elapsed_time) + ' seconds.')

            time.sleep(10)

            elapsed_time += 10

            # Create a list to hold the Bool job complete flags
            job_complete_flags = []

            # Iterate over each job id string.
            for job_id in self._job_ids:

                print("Checking job status:")

                # TODO: Handle job completion gracefully.

                # job_id = str(job_id)

                # Issue qstat command to get job status.
                p = subprocess.Popen('qstat -r ' + job_id,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     shell=True)
                print('job_id:', job_id)

                # Get the subprocess output from qstat.
                output = p.communicate()

                # print('output:', output)

                # Compute the job completion flag.
                try:

                    # Read the qstat out, parse the state, and conv to Boolean.
                    # job_complete = output[0].split()[-2] == 'E'
                    output = output[0].split()[-2]
                    output = output.decode('utf-8')
                    job_complete = output == 'E'

                    if output == 'E':
                        print('E -   Job is exiting after having run.')
                    elif output == 'H':
                        print('H -   Job is held.')
                    elif output == 'Q':
                        print('Q -   job is queued, eligible to run or  be routed.')
                    elif output == 'R':
                        print('R -   job is running.')
                    elif output == 'T':
                        print('T -   job is being moved to new location.')
                    elif output == 'W':
                        print('W -   job is waiting for its execution time to be reached.')
                    elif output == 'S':
                        print('S -   job is suspend.')
                    else:
                        print('Unknown message:', output)


                except:

                    job_complete = True

                # Print a diagnostic.
                print('Job ' +
                      job_id +
                      ' complete? ' +
                      str(job_complete) +
                      '.')

                # Append the completion flag.
                job_complete_flags.append(job_complete)

                # If the job is complete...
                if job_complete:

                    # ...clear it form the queue.
                    p = subprocess.Popen('qdel -Wforce ' + job_id,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         shell=True)

            # AND the job complete flags together.
            jobs_complete = (all(job_complete_flags))

            # Check if we've reached timeout.
            timeout = (elapsed_time > max_runtime)

            # Open a csv for writeout.
            with open(log_dir + '/' + log_filename, 'w') as csvfile:

                # Join the parameter labels and response labels, making a header.
                # headers = self.get_parameter_labels() + response_labels
                headers = ['logfile:','total_loss:']

                # Open a writer and write the header.
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(headers)

                # Iterate over each experimental mapping and write out.
                for (experimental_config,
                     output_dir,
                     output_filename) in self._input_output_maps:

                    input_row = []

                    # experimental_config is the --pipeline_config_path string:
                    # config_file_path = config_file_path.split(' ')[1]
                    # print('experimental_config:',experimental_config)
                    config_file_name = experimental_config.split('/')[-1]
                    # print('config_file_name:',config_file_name)

                    input_row.append(config_file_name)

                    output_file = output_dir + output_filename

                    # print('output_file:', output_file)

                    # Check if the output file has been written.
                    if os.path.exists(output_file):

                        with open(output_file, 'rb') as f:

                            reader = csv.reader(f)

                            for output_row in reader:

                                csvwriter.writerow(input_row + output_row)

                    else:

                        print("output filename not found: " + output_file)

            print("-----------------")


def example_usage(FLAGS):

    # Clear and remake the log directory.
    # if tf.gfile.Exists(FLAGS.log_dir):

    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)

    # tf.gfile.MakeDirs(FLAGS.log_dir)

    # Instantiate an experiment.
    exp = SatnetClusterExperiment()

    # Set the number of reps for each config.
    # exp.set_rep_count(12)

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

    example_usage(FLAGS)

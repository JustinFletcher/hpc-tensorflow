
#!/usr/bin/python
# Example PBS cluster job submission in Python

import os
import csv
import time
import random
import argparse
import itertools
import subprocess

# TODO: Remove dependence on TF
import tensorflow as tf


class ClusterExperiment(object):

    def __init__(self):

        self.experimental_configs = []
        self.independent_designs = []
        self.coupled_designs = []
        self.parameter_labels = []
        self._input_output_maps = []
        self._job_ids = []

    def add_design(self, flag, value_list):

        self.independent_designs.append((flag, value_list))

    def add_coupled_design(self, coupled_design):

        self.coupled_designs.append(coupled_design)

    def set_rep_count(self, num_reps):

        self.add_design('rep_num', range(num_reps))

    def get_configs(self):

        # Translate the design structure into flag strings.
        exp_flag_strings = [['--' + f + '=' + str(v) for v in r]
                            for (f, r) in self.independent_designs]

        # Produce the Cartesian set of configurations.
        indep_experimental_configs = list(itertools.product(*exp_flag_strings))

        # Initialize a list to store coupled configurations.
        coupled_configs = []

        # Scope this variable higher due to write-out coupling.
        coupled_flag_strs = []

        for coupled_design in self.coupled_designs:

            for d in coupled_design:

                coupled_flag_strs = [['--' + f + '=' + str(v) for v in r]
                                     for (f, r) in d]

                coupled_configs += list(itertools.product(*coupled_flag_strs))

        # Initialize empty experimental configs list...
        experimental_configs = []

        # ...and if there are coupled configs...
        if coupled_configs:

            # ...iterate over each independent config...
            for e in indep_experimental_configs:

                    # ...then for each coupled config...
                    for c in coupled_configs:

                        # ...join the coupled config to the independent one.
                        experimental_configs.append(e + tuple(c))

        # Otherwise, ....
        else:

            # ...just pass the independent experiments through.
            experimental_configs = indep_experimental_configs

        return(experimental_configs)

    def get_parameter_labels(self):

        parameter_labels = []

        for (f, _) in self.independent_designs:

            parameter_labels.append(f)

        for coupled_design in self.coupled_designs:

            coupled_flag_strs = []

            for d in coupled_design:

                coupled_flag_strs = [f for (f, _) in d]

            parameter_labels += coupled_flag_strs

        return(parameter_labels)

    def launch_experiment(self,
                          exp_filename,
                          log_dir,
                          account_str,
                          queue_str,
                          module_str,
                          manager='pbs',
                          shuffle_job_order=True):

        experimental_configs = self.get_configs()

        if shuffle_job_order:

            # Shuffle the submission order of configs to avoid asymetries.
            random.shuffle(experimental_configs)

        # Iterate over each experimental configuration, launching jobs.
        for i, experimental_config in enumerate(experimental_configs):

            if manager == 'pbs':

                # Some prints for easy debug.
                # print("-----------experimental_config---------")
                # print(experimental_config)
                # print("---------------------------------------")

                # Use subproces to command qsub to submit a job.
                p = subprocess.Popen('qsub',
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     shell=True)

                # Customize your options here.
                job_name = "dist_ex_%d" % i
                walltime = "72:00:00"
                select = "1:ncpus=20:mpiprocs=20"
                command = "python " + exp_filename

                # Iterate over flag strings, building the command.
                for flag in experimental_config:

                    command += ' ' + flag

                # Add a final flag modifying the log filename to be unique.
                log_filename = 'templog' + str(i)

                temp_log_dir = log_dir + 'templog' + str(i) + '/'

                command += ' --log_dir=' + temp_log_dir

                # Add the logfile to the command.
                command += ' --log_filename=' + log_filename

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
                # module load anaconda2/5.0.1 gcc/5.3.0 cudnn/6.0
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

                # Send job_string to qsub, returning a job ID in bytes.
                job_id = p.communicate(job_string)[0]

                print(job_id)
                # Parse the bytes to an ID string.
                # job_id = str(job_id)

                # Append the job ID.
                self._job_ids.append(job_id)

                print(self._job_ids)
                time.sleep(0.1)

            else:

                print("Unknown manager supplied to launch_experiment().")
                exit()

# qsub -I -A MHPCC96670DA1 -q standard -l select=1:ncpus=20:mpiprocs=20 -l walltime=1:00:00

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

                print("is this thing on?")

                # TODO: Handle job completion gracefully.

                # Issue qstat command to get job status.
                p = subprocess.Popen('qstat -r ' + job_id,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     shell=True)
                print(job_id)

                # Get the subprocess output from qstat.
                output = p.communicate()

                # Compute the job completion flag.
                try:

                    # Read the qstat out, parse the state, and conv to Boolean.
                    job_complete = output[0].split()[-2] == 'E'

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

            # And the job complete flags together.
            jobs_complete = (all(job_complete_flags))

            # Check if we've reached timeout.
            timeout = (elapsed_time > max_runtime)

            # Open a csv for writeout.
            with open(log_dir + '/' + log_filename, 'w') as csvfile:

                # Join parameter labels and respons labels, making a header.
                headers = self.get_parameter_labels() + response_labels

                # Open a writer and write the header.
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(headers)

                # Iterate over each eperimental mapping and write out.
                for (input_flags,
                     output_dir,
                     output_filename) in self._input_output_maps:

                    input_row = []

                    # Process the flags into output values.
                    for flag in input_flags:

                        flag_val = flag.split('=')[1]

                        input_row.append(flag_val)

                    output_file = output_dir + output_filename

                    # Check if the output file has been written.
                    if os.path.exists(output_file):

                        with open(output_file, 'rb') as f:

                            reader = csv.reader(f)

                            for output_row in reader:

                                csvwriter.writerow(input_row + output_row)

                    else:

                        print("output filename not found: " + output_filename)

            print("-----------------")


def example_usage(FLAGS):

    # Clear and remake the log directory.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Instantiate an experiment.
    exp = ClusterExperiment()

    # Set the number of reps for each config.
    exp.set_rep_count(12)

    # Set independent parameters.
    exp.add_design('train_batch_size', [128, 256])
    exp.add_design('train_enqueue_threads', [32, 64])
    exp.add_design('learning_rate', [0.0001])
    exp.add_design('max_steps', [10000])
    exp.add_design('test_interval', [100])
    exp.add_design('pause_time', [10])

    # Launch the experiment.
    exp.launch_experiment(exp_filename=FLAGS.experiment_py_file,
                          log_dir=FLAGS.log_dir,
                          account_str='MHPCC96670DA1',
                          queue_str='standard',
                          module_str='anaconda2/5.0.1 gcc/5.3.0 cudnn/6.0',
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
                        default='../log/cluster_experiment/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='cluster_experiment.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=3600000,
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--experiment_py_file', type=str,
                        default='~/hpc-tensorflow/cluster_experiment_example.py',
                        help='Number of seconds to run before giving up.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    example_usage(FLAGS)

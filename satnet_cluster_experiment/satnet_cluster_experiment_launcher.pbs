#!/bin/bash
## Walltime in hours:minutes:seconds
#PBS -l walltime=2:00:00
## -o specifies output file
#PBS -o ~/log/queue_exhaustion.out
## -e specifies error file
#PBS -e ~/log/queue_exhaustion.error
## Nodes, Processors, CPUs (processors and CPUs should always match)
#PBS -l select=1:mpiprocs=20:ncpus=20
## Enter the proper queue
#PBS -q standard
#PBS -A MHPCC96650DE1
module load anaconda3/5.0.1 tensorflow/1.8.0
## module unload anaconda2/5.0.1
cd /gpfs/projects/ml/hpc-tensorflow/satnet_cluster_experiment/
python satnet_cluster_experiment_launcher.py
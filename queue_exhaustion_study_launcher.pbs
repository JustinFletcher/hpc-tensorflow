#!/bin/bash
## Walltime in hours:minutes:seconds
#PBS -l walltime=48:00:00
## -o specifies output file
#PBS -o ~/log/queue_exhaustion.out
## -e specifies error file
#PBS -e ~/log/queue_exhaustion.error
## Nodes, Processors, CPUs (processors and CPUs should always match)
#PBS -l select=1:mpiprocs=20:ncpus=20
## Enter the proper queue
#PBS -q standard
#PBS -A MHPCC96650DE1
cp -R ~/data $WORKDIR
module load anaconda3/5.0.1 tensorflow
cd ~/hpc-tensorflow
python queue_exhaustion_study_launcher.py
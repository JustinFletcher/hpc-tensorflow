#!/bin/bash
## Walltime in hours:minutes:seconds
#PBS -l walltime=1:00:00
## -o specifies output file
#PBS -o ~/log/queue_exhaustion.out
## -e specifies error file
#PBS -e ~/log/queue_exhaustion.error
## Nodes, Processors, CPUs (processors and CPUs should always match)
#PBS -l select=1:mpiprocs=20:ncpus=20
## Enter the proper queue
#PBS -q standard
#PBS -A MHPCC96650DE1
# module load anaconda2/5.0.1 gcc/5.3.0 cudnn/6.0
# module load tensorflow/1.4.0
module load anaconda3/5.0.1 tensorflow
cd /gpfs/home/fletch/hpc-tensorflow/astronet_faster_rcnn_study/
python astronet_faster-rcnn_resnet101_cluster_experiment_launcher.py
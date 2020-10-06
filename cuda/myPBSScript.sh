#!/bin/bash

# Which Queue to use, DO NOT CHANGE #
#PBS -q GPUq

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:01:00

# How many nodes and tasks per node,  1 nodes with 8 tasks(threads)#
#PBS -lselect=1:ncpus=10:ompthreads=10:ngpus=2 -lplace=excl

# JobName #
#PBS -N cudaJob

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
nvprof ./game_of_life

# profile executable #
#nvprof ./simple_add1GPUOmp

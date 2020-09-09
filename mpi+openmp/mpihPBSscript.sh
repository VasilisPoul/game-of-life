#!/bin/bash

# Max VM size #
#PBS -l pvmem=2G

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:01:00

# How many nodes, cpus/node, mpiprocs/node and threds/mpiprocess 
# Example 2 nodes with 8 cpus, 2 mpirocs and 4 threads
#PBS -l select=8:ncpus=8:mpiprocs=4:ompthreads=2

# Only this job uses the chosen nodes
#PBS -l place=excl

# Which Queue to use, DO NOT CHANGE #
#PBS -q workq

# JobName #
#PBS -N a

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun mpiH_trapDemo.x

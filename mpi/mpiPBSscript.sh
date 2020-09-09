#!/bin/bash

# Max VM size #
#PBS -l pvmem=2G

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:01:00

# Only this job uses the chosen nodes
#PBS -l place=excl

# Which Queue to use, DO NOT CHANGE #
#PBS -q workq

# JobName #
#PBS -N golJob

#Change Working directory to SUBMIT director
cd $PBS_O_WORKDIR

# Run executable #
mpirun -np $proc game_of_life -i $inputFilePath -f $outputFolder -r $rows -c $cols 

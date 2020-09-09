#!/bin/bash

#load module
module load mpiP

#compile
mpicc -O3 -g game_of_life.c main.c mpi.c -L$MPIP_DIR/lib -lmpiP -lm -Wall -o game_of_life

#run
rm *.mpiP golJob.* core.* *.x
rows=32
cols=32
inputFilePath=generations/row/input.txt
outputFolder=generations/row/
rm -f generations/row/*
rm -f generations/boxes/*
python3 scripts/block.py $rows $inputFilePath

select=2
ncpus=8
mpiprocs=16
echo "Setting select to "$select
echo "Setting ncpus to "$ncpus
echo "Setting mpiprocs to "$mpiprocs

ID=$(qsub -l select=$select:ncpus=$ncpus:mpiprocs=$mpiprocs -v inputFilePath=$inputFilePath,outputFolder=$outputFolder,proc=$mpiprocs,rows=$rows,cols=$cols mpiPBSscript.sh | sed -e s/"\..*"//)
while [[ ! -z $(qstat | grep argo082) ]]; do
  sleep 0.5
done

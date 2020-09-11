#!/bin/bash

#load module
module load mpiP

#compile
mpicc -O3 -g game_of_life.c main.c mpi.c -L$MPIP_DIR/lib -lmpiP -lm -Wall -o game_of_life

#run
rm *.mpiP golJob.* core.* *.x

#config dimensions
rows=320
cols=320

ncpus=8

select=3
mpiprocs=3
np=9


#
inputFilePath=generations/row/input.txt
outputFolder=generations/row/
rm -f generations/row/*
rm -f generations/boxes/*
python3 scripts/block.py $rows $inputFilePath

echo "Setting select to "$select
echo "Setting ncpus to "$ncpus
echo "Setting mpiprocs to "$mpiprocs
echo "Setting np to "$np

ID=$(qsub -l select=$select:ncpus=$ncpus:mpiprocs=$mpiprocs -N golJob -v inputFilePath=$inputFilePath,outputFolder=$outputFolder,proc=$np,rows=$rows,cols=$cols mpiPBSscript.sh | sed -e s/"\..*"//)
while [[ ! -z $(qstat | grep argo082) ]]; do
  sleep 0.5
done

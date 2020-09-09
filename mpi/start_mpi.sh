#!/bin/bash

rm *.mpiP golJob.* core.* *.x

module load mpiP

rm -f generations/row/*
rm -f generations/boxes/*
python3 scripts/block.py 32 generations/row/input.txt

mpicc -O3 -g game_of_life.c main.c mpi.c -L$MPIP_DIR/lib -lmpiP -lm -Wall -o game_of_life.x

select=2
ncpus=8
mpiprocs=16
echo "Setting select to "$select
echo "Setting ncpus to "$ncpus
echo "Setting mpiprocs to "$mpiprocs

ID=$(qsub -l select=$select:ncpus=$ncpus:mpiprocs=$mpiprocs -v proc=$mpiprocs mpiPBSscript.sh | sed -e s/"\..*"//)
while [[ ! -z $(qstat | grep argo082) ]]; do
  sleep 0.5
done

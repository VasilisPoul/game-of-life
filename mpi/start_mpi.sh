#!/bin/bash

rm *.mpiP a.* core.*

module load mpiP

mpicc -O3 -g game_of_life.c main.c mpi.c -L$MPIP_DIR/lib -lmpiP -lbfd -lunwind -lm -Wall -o game_of_life.x

select=1
ncpus=1
mpiprocs=1
echo "Setting select to "$select
echo "Setting ncpus to "$ncpus
echo "Setting mpiprocs to "$mpiprocs

ID=$(qsub -l select=$select:ncpus=$ncpus:mpiprocs=$mpiprocs -v proc=$mpiprocs mpiPBSscript.sh | sed -e s/"\..*"//)
while [[ ! -z $(qstat | grep argo082) ]]; do
  sleep 0.5
done

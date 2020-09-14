#!/bin/sh

# shellcheck disable=SC2045
for file in $(ls /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/row/)
do
  python3 /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/scripts/boxes.py 320 /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/row/$file /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/boxes/$file
done

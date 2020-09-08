#!/bin/sh

# shellcheck disable=SC2045
for file in $(ls /home/vasilis/projects/game-of-life/mpi+openmp/generations/row/)
do
  python3 /home/vasilis/projects/game-of-life/scripts/boxes.py 960 /home/vasilis/projects/game-of-life/mpi+openmp/generations/row/$file /home/vasilis/projects/game-of-life/mpi+openmp/generations/boxes/$file
done

#!/bin/sh

# shellcheck disable=SC2045
for file in $(ls /home/msi/projects/CLionProjects/game-of-life/mpi/generations/row/)
do
  python3 /home/msi/projects/CLionProjects/game-of-life/scripts/boxes.py 32 /home/msi/projects/CLionProjects/game-of-life/mpi/generations/row/$file /home/msi/projects/CLionProjects/game-of-life/mpi/generations/boxes/$file
done

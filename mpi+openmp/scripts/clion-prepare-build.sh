#!/bin/sh
rm -f /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/row/steps-*
rm -f /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/boxes/*
rm -f /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/row/input.txt
python3 /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/scripts/block.py 320 /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/row/input.txt

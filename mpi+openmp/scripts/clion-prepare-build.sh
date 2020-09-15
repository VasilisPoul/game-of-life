#!/bin/sh
rm -f /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/row/*
rm -f /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/boxes/*
python3 /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/scripts/block.py 1280 /home/msi/projects/CLionProjects/game-of-life/mpi+openmp/generations/row/input.txt

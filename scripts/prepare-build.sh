#!/bin/sh
rm -f /home/vasilis/projects/game-of-life/mpi+openmp/generations/row/*
rm -f /home/vasilis/projects/game-of-life/mpi+openmp/generations/boxes/*
python3 /home/vasilis/projects/game-of-life/scripts/block.py 960 /home/vasilis/projects/game-of-life/mpi+openmp/generations/row/input.txt

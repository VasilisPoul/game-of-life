#!/bin/bash

rm  cudaJob.* times.*

for gpus in 1 2; do
  for dims in 320 640 1280 2560 5120; do
    TF="times_"$gpus"gpus.txt"
    for m in 4 8 16 32 64; do
      sed -e "s/\#define N .*/\#define N "$dims" /" \
      -e "s/\#define M .*/\#define M "$m" /" \
      -e "s/\#define FILE_NAME .*\"/\#define FILE_NAME \"test-files\/"$dims"x"$dims".txt\"/" \
      main.cu  >main.cu.new
      mv main.cu.new main.cu
      nvcc -o game_of_life main.cu 

      sed -s "s/#PBS -lselect=1:ncpus=10:ompthreads=10:ngpus=[0-9]/#PBS -lselect=1:ncpus=10:ompthreads=10:ngpus="$gpus"/" \
      myPBSScript.sh  >myPBSScript.sh.new
      mv myPBSScript.sh.new myPBSScript.sh

      ID=$(qsub myPBSScript.sh | sed -e s/"\..*"//)

      sp="/-\|"
      echo -n ' '
      echo "Waiting job '"$ID"' with dimensions "$dims"x"$dims" and threads/block: "$m" ..."
      while [[ ! -z $(qstat | grep argo082) ]]; do
        printf "\b${sp:i++%${#sp}:1}"   
        sleep 0.3
      done
      sed -e "s/time_spent=/"$dims"x"$dims" M=$m: /" cudaJob.o$ID >>$TF
    done
    echo >>$TF
  done
done



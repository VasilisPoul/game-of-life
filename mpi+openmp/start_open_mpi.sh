#!/bin/bash

#load module
module load mpiP
module load openmpi3

#compile
mpicc -O3 -g -fopenmp game_of_life.c main.c mpi.c -L$MPIP_DIR/lib -lmpiP -lm -Wall -o game_of_life

#rm
rm *.mpiP golJob*.* *.x times.* speedup.* efficiency.*
rm -f generations/row/*
rm -f generations/boxes/*

#
inputFilePath=generations/row/input.txt
outputFolder=generations/row/

rows=320
cols=320

ompthreads=2
ncpus=8

for i in {1..1}; do

  #config dimensions
  rows=$(python -c "print("$rows" * "$i")")
  cols=$(python -c "print("$cols" * "$i")")

  echo "Generating input file."
  python3 scripts/block.py $rows $inputFilePath

  TF="times."$rows"x"$cols".txt"

  processes=(1)

  for j in ${processes[@]}; do 

      case "$j" in
    1 )
      select=1
      mpiprocs=1
      ;;&
    4 )
      select=1
      mpiprocs=4      
      ;;&   
    16 )
      select=2
      mpiprocs=8       
      ;;&
    64 )
      select=8
      mpiprocs=8      
      ;;&
    esac
  
    np=$j

    echo "Run with "$np" processes, nodes: "$select", cpus: "$ncpus", processes per node: "$mpiprocs

    ID=$(qsub -l select=$select:ncpus=$ncpus:mpiprocs=$mpiprocs:ompthreads=$ompthreads -N golJob_$j"_"$rows -v inputFilePath=$inputFilePath,outputFolder=$outputFolder,proc=$np,rows=$rows,cols=$cols mpihPBSscript.sh | sed -e s/"\..*"//)
    
    i=1
    sp="/-\|"
    echo -n ' '
    echo "Waiting... Processes: "$j" ... Dimensions: "$rows" x "$cols
    while [[ ! -z $(qstat | grep argo082) ]]; do
      printf "\b${sp:i++%${#sp}:1}"   
      sleep 0.3
    done
    (grep "Steps" < "golJob_"$j"_"$rows".o"$ID ) | sed -e "s/Steps: [0-9]*, Max time: /"$j": /" >>$TF
  done

done

# #speedup & efficiency
for tf in times.*.txt; do
  head=$(head -n 1 "$tf")
  TS=$(echo "$head" | sed -e "s/[0-9]*: //")
  echo "TS="$TS
  sf=$(printf $tf | sed -e "s/times/speedup/")
  ef=$(printf $tf | sed -e "s/times/efficiency/")
  echo $sf, $ef
  touch $sf
  touch $ef
  P=1
  while read line; do
  TP=$(echo "$line" | sed -e "s/[0-9]*: //")
  ps=$(echo "$line" | grep -G -o "[0-9]*: ")
  S=$(python -c "print("$TS" / "$TP")")
  E=$(python -c "print("$S" / "$P")")
  P=$((P * 2))
  echo $ps$S >>"$sf"
  echo $ps$E >>"$ef"
  done < $tf
done

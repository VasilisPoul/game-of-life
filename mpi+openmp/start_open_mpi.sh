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

threads=(1 2 4 8 16)

ncpus=8
ompthreads=0
for t in ${threads[@]}; do
  for i in {1..1}; do

    #config dimensions
    rows=$(python -c "print("$rows" * "$i")")
    cols=$(python -c "print("$cols" * "$i")")

    echo "Generating input file."
    python3 scripts/block.py $rows $inputFilePath

    TF="times."$rows"x"$cols".txt"
    processes=(1 4 16 64)

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

      ID=$(qsub -l select=$select:ncpus=$ncpus:mpiprocs=$mpiprocs:ompthreads=$t -N golJob_p$j"_t"$t"_dims"$rows -v inputFilePath=$inputFilePath,outputFolder=$outputFolder,proc=$np,rows=$rows,cols=$cols mpihPBSscript.sh | sed -e s/"\..*"//)
      
      i=1
      sp="/-\|"
      echo -n ' '
      echo "Waiting... Processes: "$j" ... Threads: "$t" ... Dimensions: "$rows" x "$cols
      while [[ ! -z $(qstat | grep argo021) ]]; do
        printf "\b${sp:i++%${#sp}:1}"
        sleep 0.3
      done
      (grep "Steps" < "golJob_p"$j"_t"$t"_dims"$rows".o"$ID ) | sed -e "s/Steps: [0-9]*, Max time: /Processes "$j" Threads "$t": /" >>$TF
    done

  done
done


#speedup & efficiency
for tf in times.*.txt; do
  i=0
  TS=0
  sf=0
  ef=0
  P=0
  while read line; do
  if ! ((i % 4)); then
    sed=$( sed -n $((i+1)),$((i+1))p "$tf")
    TS=$(echo "$sed" | sed -e "s/Processes [0-9]* Threads [0-9]*: //")
    sf=$(printf $tf | sed -e "s/times/speedup/")
    ef=$(printf $tf | sed -e "s/times/efficiency/")
    touch $sf
    touch $ef
    echo >>"$sf"
    echo >>"$ef"
    P=1
  fi
  TP=$(echo "$line" | sed -e "s/Processes [0-9]* Threads [0-9]*: //")
  ps=$(echo "$line" | grep -G -o "Processes [0-9]* Threads [0-9]*: ")
  S=$(python -c "print("$TS" / "$TP")")
  E=$(python -c "print("$S" / "$P")")
  P=$((P * 2))
  echo $ps$S >>"$sf"
  echo $ps$E >>"$ef"
  i=$((i+1))
  done < $tf
done

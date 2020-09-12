#!/bin/bash

#load module
module load mpiP

#compile
mpicc -O3 -g game_of_life.c main.c mpi.c -L$MPIP_DIR/lib -lmpiP -lm -Wall -o game_of_life

#rm
rm *.mpiP golJob*.* core.* *.x times.*
rm -f generations/row/*
rm -f generations/boxes/*

#
inputFilePath=generations/row/input.txt
outputFolder=generations/row/

rows=320
cols=320
for i in {1..4}; do

  #config dimensions
  rows=$(python -c "print("$rows" * "$i")")
  cols=$(python -c "print("$cols" * "$i")")

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
  
    ncpus=8
    np=$j

    echo "Setting select to "$select
    echo "Setting ncpus to "$ncpus
    echo "Setting mpiprocs to "$mpiprocs
    echo "Setting np to "$np

    ID=$(qsub -l select=$select:ncpus=$ncpus:mpiprocs=$mpiprocs -N golJob_$j"_"$rows -v inputFilePath=$inputFilePath,outputFolder=$outputFolder,proc=$np,rows=$rows,cols=$cols mpiPBSscript.sh | sed -e s/"\..*"//)
    while [[ ! -z $(qstat | grep argo082) ]]; do
      echo "waiting... "$j" ... "$rows
      sleep 0.5
    done
    (grep "Steps" < "golJob_"$j"_"$rows".o"$ID ) | sed -e "s/Steps: [0-9]*, Max time: /"$j": /" >>$TF
  done

done


#speedup & efficiency
for tf in times.*.txt; do


  TS=$(head -n 1 "$tf" | sed -e "s/[0-9]*: //")
  echo "TS="$TS
  sf=$(printf $tf | sed -e "s/times/speedup/")
  ef=$(printf $tf | sed -e "s/times/efficiency/")
  echo $sf, $ef
  touch $sf
  touch $ef
  P=1
  x=$(tail -n +1 $tf)
  # processes=$(x | sed -e "s/[0-9]*: //")
  for TP in $(x sed -e "s/[0-9]*: //"); do
    echo "TP"=$TP
    S=$(python -c "print("$TS" / "$TP")")
    E=$(python -c "print("$S" / "$P")")
    P=$((P * 2))
    echo $S >>"$sf"
    echo $E >>"$ef"
  done
done
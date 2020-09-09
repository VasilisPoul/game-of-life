#!/bin/bash

SCRIPTNAME=mpihPBSscript

rm *.mpiP myJob.* core.* efficiency.* speedup.* times.*

for i in {24..30..2}; do
 TRAPS=$((2 ** i))
 TF="times."$TRAPS".txt"
 sed -e "s/\#define nTraps .*/\#define nTraps "$TRAPS" /" mpiH_trapDemo.c >mpiH_trapDemo.c.new
 mv mpiH_trapDemo.c.new mpiH_trapDemo.c
 echo "Recompiling mpiH_trapDemo.c for "$TRAPS" trapezoids..."
 module load mpiP
 module load openmpi3
 mpicc -O3 -fopenmp mpiH_trapDemo.c -L$MPIP_DIR/lib -lmpiP -o mpiH_trapDemo.x
 for j in {0..6}; do
   x=$((2 ** j))
   echo "Setting nodes to "$x
   if [ $x -gt 4 ]; then
     sed -e "s/\#PBS -l select=[0-9]*:ppn=8/\#PBS -l select="$((x / 8))":ppn=8/" -e "s/mpirun.* mpiH_trapDemo.x/mpirun mpiH_trapDemo.x/" $SCRIPTNAME"".sh >$SCRIPTNAME"".sh.new
     mv $SCRIPTNAME"".sh.new $SCRIPTNAME"".sh
   else
     sed -e "s/\#PBS -l select=[0-9]*:ppn=8/\#PBS -l select=1:ppn=8/" -e "s/mpirun.* mpiH_trapDemo.x/mpirun -np "$x" mpiH_trapDemo.x/" $SCRIPTNAME"".sh >$SCRIPTNAME"".sh.new
     mv $SCRIPTNAME"".sh.new $SCRIPTNAME"".sh
   fi
   ID=$(qsub $SCRIPTNAME"".sh | sed -e s/"\..*"//)
   while [[ ! -z $(qstat | grep argo082) ]]; do
     sleep 0.5
   done
   (grep "Time : " < "a.o"$ID ) | sed -e s/"Time : "// >>$TF
 done
done

#speedup & efficiency
for tf in times.*.txt; do
  TS=$(head -n 1 "$tf")
  echo "TS="$TS
  sf=$(printf $tf | sed -e "s/times/speedup/")
  ef=$(printf $tf | sed -e "s/times/efficiency/")
  echo $sf, $ef
  touch $sf
  touch $ef
  P=1
  for TP in $(tail -n +1 $tf); do
    echo "TP"=$TP
    S=$(python -c "print("$TS" / "$TP")")
    E=$(python -c "print("$S" / "$P")")
    P=$((P * 2))
    echo $S >>"$sf"
    echo $E >>"$ef"
  done
done
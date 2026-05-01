#!/bin/bash

N=10
Re=1000
OUT_DIR="../N${N}_Re${Re}"
OUT_FILE="${OUT_DIR}/results.out"

rm -rf $OUT_DIR
mkdir $OUT_DIR

python solver.py -n $N -o $OUT_DIR -r $Re -d 0.001 -t 1000 --plot_every 100 >> $OUT_FILE
python postproc.py -n $N -i $OUT_DIR -r $Re -t 0.5 >> $OUT_FILE
python postproc.py -n $N -i $OUT_DIR -r $Re -t 1.0 >> $OUT_FILE
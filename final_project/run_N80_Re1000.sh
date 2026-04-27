#!/bin/bash

N=80
Re=1000
OUT_DIR="N${N}_Re${Re}"

rm -rf $OUT_DIR
mkdir $OUT_DIR

python solver.py -n $N -o $OUT_DIR -r $Re
python postproc.py -n $N -i $OUT_DIR -r $Re -t 0.5
python postproc.py -n $N -i $OUT_DIR -r $Re -t 1.0
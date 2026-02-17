#!/bin/bash

RESULTS_DIR=results_hw2c
mkdir -p $RESULTS_DIR

OUTFILE=$RESULTS_DIR/output.csv
rm -f $OUTFILE
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $OUTFILE

python poisson_solver.py -n 11 21 41 -m jacobi -a 1.0 1.0 1.0 -o $RESULTS_DIR >> $OUTFILE
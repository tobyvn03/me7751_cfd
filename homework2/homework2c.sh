#!/bin/bash

RESULTS_DIR=results_hw2c
mkdir -p $RESULTS_DIR

OUTFILE=$RESULTS_DIR/output.csv
rm -f $OUTFILE
echo "N, Method, Iterations, Time (seconds), Error (MAE)" >> $OUTFILE

python poisson_solver.py -n 11 21 41 -m jacobi -k 1000 2000 6000 -t 1e-2 -o $RESULTS_DIR >> $OUTFILE

python plot_csv.py -i $OUTFILE -x 0 -y 4 -n 5 -t "Mean Absolute Error vs. Grid Size" -o "$RESULTS_DIR/errors.png" -l -L
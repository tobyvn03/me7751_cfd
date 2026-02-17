#!/bin/bash

RESULTS_DIR=results_hw2c
mkdir -p $RESULTS_DIR

OUTFILE=$RESULTS_DIR/output.csv
rm -f $OUTFILE
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha" >> $OUTFILE

python poisson_solver.py -n 11 21 41 -m jacobi -a 1.0 1.0 1.0 -o $RESULTS_DIR >> $OUTFILE

python plot_csv.py -i $OUTFILE -x 0 -y 4 -n 5 -t "Mean Absolute Error vs. Grid Size" -o "$RESULTS_DIR/errors.png" -l -L
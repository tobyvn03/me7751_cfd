#!/bin/bash

OUTFILE=results/homework2c_output.csv
rm -f $OUTFILE
echo "N, Method, Iterations, Time (seconds), Error (MAE)" >> $OUTFILE

python poisson_solver.py -n 10 20 40 -m jacobi -k 1000 2000 6000 -t 1e-6 >> $OUTFILE
#!/bin/bash

OUTFILE=results/homework2c_output.csv
rm -f $OUTFILE
echo "N, Method, Iterations, Time (seconds), Error (MAE)" >> $OUTFILE

python poisson_solver.py -n 10 20 40 80 160 -m jacobi -k 1000 2000 6000 24000 96000 -t 1e-6 >> $OUTFILE
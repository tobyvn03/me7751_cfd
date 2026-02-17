#!/bin/bash

RESULTS_DIR=results_hw2d
mkdir -p $RESULTS_DIR

OUTFILE=$RESULTS_DIR/output_no_ghost.csv
OUTFILEG=$RESULTS_DIR/output_with_ghost.csv
rm -f $OUTFILE
rm -f $OUTFILEG
echo "N, Method, Iterations, Time (seconds), Error (MAE)" >> $OUTFILE
echo "N, Method, Iterations, Time (seconds), Error (MAE)" >> $OUTFILEG

python poisson_solver.py -n 10 20 40 80 160 -m jacobi -k 1000 2000 6000 24000 96000 -t 1e-6 -o $RESULTS_DIR >> $OUTFILE
python poisson_solver.py -n 10 20 40 80 160 -m jacobi -k 1000 2000 6000 24000 96000 -t 1e-6 -g -o $RESULTS_DIR >> $OUTFILEG

python plot_csv.py -i $OUTFILE -x 0 -y 4 -n 5 -t "Mean Absolute Error vs. Grid Size: No Ghost Cells" -o "$RESULTS_DIR/errors_no_ghost.png" -l
python plot_csv.py -i $OUTFILEG -x 0 -y 4 -n 5 -t "Mean Absolute Error vs. Grid Size: Using Ghost Cells" -o "$RESULTS_DIR/errors_with_ghost.png" -l
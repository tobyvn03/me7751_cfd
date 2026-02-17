#!/bin/bash

RESULTS_DIR=results_hw2d
mkdir -p $RESULTS_DIR

OUTFILE=$RESULTS_DIR/without_ghost_cells.csv
GHOST=$RESULTS_DIR/with_ghost_cells.csv
rm -f $OUTFILE
rm -f $GHOST
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $OUTFILE
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $GHOST

python poisson_solver.py -n 161 81 41 21 11 -m jacobi -a 1.0 1.0 1.0 1.0 1.0 >> $OUTFILE
python poisson_solver.py -n 161 81 41 21 11 -m jacobi -a 1.0 1.0 1.0 1.0 1.0 -g >> $GHOST

python plot_csv.py -i $OUTFILE $GHOST -x 6 -y 4 -t "Mean Absolute Error vs. Grid Size" -o "$RESULTS_DIR/errors.png" -L -l -p 1.1 1.1
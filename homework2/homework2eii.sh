#!/bin/bash

RESULTS_DIR=results_hw2e
mkdir -p $RESULTS_DIR

ALPHA=$RESULTS_DIR/explore_alpha.csv
rm -f $ALPHA
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $ALPHA

python poisson_solver.py -n 41 41 41 41 41 41 41 41 41 41 -m sor -a 1.85 1.86 1.87 1.88 1.89 1.9 1.91 1.92 1.93 1.94 >> $ALPHA

python plot_csv.py -i $ALPHA -x 5 -y 2 -t "Relaxation Factor vs. Iteration Count" -o "$RESULTS_DIR/explore_alpha.png"
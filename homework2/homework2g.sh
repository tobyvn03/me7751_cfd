#!/bin/bash

RESULTS_DIR=results_hw2g
mkdir -p $RESULTS_DIR

MG=$RESULTS_DIR/multi_grid.csv
rm -f $MG
# echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $MG
echo "Cycle, Iterations, Residual" >> $MG

python poisson_solver.py -n 9 -m multi-grid -a 1.0 -o $RESULTS_DIR -v >> $MG

# python plot_csv.py -i $OUTFILE -x 0 -y 3 -t "Computation Time vs. Grid Size" -o "$RESULTS_DIR/times.png" -L -l -p 4
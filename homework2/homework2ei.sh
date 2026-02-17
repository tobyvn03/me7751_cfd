#!/bin/bash

RESULTS_DIR=results_hw2e
mkdir -p $RESULTS_DIR

JACOBI=$RESULTS_DIR/jacobi.csv
GAUSS=$RESULTS_DIR/gauss_seidel.csv
SOR=$RESULTS_DIR/sor.csv
rm -f $JACOBI
rm -f $GAUSS
rm -f $SOR
echo "Iteration, Residual" >> $JACOBI
echo "Iteration, Residual" >> $GAUSS
echo "Iteration, Residual" >> $SOR

python poisson_solver.py -n 41 -m jacobi -k 6000 -t 1e-2 -v >> $JACOBI
python poisson_solver.py -n 41 -m gauss-seidel -k 6000 -t 1e-2 -v >> $GAUSS
python poisson_solver.py -n 41 -m sor -k 6000 -t 1e-2 -a 1.2 -v >> $SOR

python plot_csv.py -i $JACOBI $GAUSS $SOR -x 0 -y 1 -t "Residuals of Each Method" -o "$RESULTS_DIR/residuals.png" -L
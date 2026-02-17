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

python poisson_solver.py -n 40 -m jacobi -k 6000 -t 1e-6 -v >> $JACOBI
python poisson_solver.py -n 40 -m gauss-seidel -k 6000 -t 1e-6 -v >> $GAUSS
python poisson_solver.py -n 40 -m sor -k 6000 -t 1e-6 -a 1.5 -v >> $SOR

python plot_csv.py -i $RESULTS_DIR/jacobi.csv $RESULTS_DIR/gauss_seidel.csv $RESULTS_DIR/sor.csv -x 0 -y 1 -t "Residuals of Each Method" -o "$RESULTS_DIR/residuals.png"
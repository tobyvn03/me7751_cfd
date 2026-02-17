#!/bin/bash

RESULTS_DIR=results_hw2f
mkdir -p $RESULTS_DIR

JACOBI=$RESULTS_DIR/jacobi.csv
GAUSS=$RESULTS_DIR/gauss_seidel.csv
SOR=$RESULTS_DIR/sor.csv
rm -f $JACOBI
rm -f $GAUSS
rm -f $SOR
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $JACOBI
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $GAUSS
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $SOR

python poisson_solver.py -n 11 21 41 81 161 -m jacobi -a 1.0 1.0 1.0 1.0 1.0 >> $JACOBI
python poisson_solver.py -n 11 21 41 81 161 -m gauss-seidel -a 1.0 1.0 1.0 1.0 1.0 >> $GAUSS
python poisson_solver.py -n 11 21 41 81 161 -m sor -a 1.92 1.92 1.92 1.92 1.92 >> $SOR

python plot_csv.py -i $JACOBI $GAUSS $SOR -x 0 -y 3 -t "Computation Time vs. Grid Size" -o "$RESULTS_DIR/times.png" -L -l -p 4
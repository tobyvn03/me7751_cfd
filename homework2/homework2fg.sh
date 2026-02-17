#!/bin/bash

RESULTS_DIR=results_hw2f
mkdir -p $RESULTS_DIR

JACOBI=$RESULTS_DIR/jacobi.csv
GAUSS=$RESULTS_DIR/gauss_seidel.csv
SOR=$RESULTS_DIR/sor.csv
MG=$RESULTS_DIR/multi_grid.csv
# rm -f $JACOBI
# rm -f $GAUSS
# rm -f $SOR
rm -f $MG
# echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $JACOBI
# echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $GAUSS
# echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $SOR
echo "N, Method, Iterations, Time (seconds), Error (MAE), Alpha, h" >> $MG

# python poisson_solver.py -n 11 21 41 81 161 -m jacobi -a 1.0 1.0 1.0 1.0 1.0 >> $JACOBI
# python poisson_solver.py -n 11 21 41 81 161 -m gauss-seidel -a 1.0 1.0 1.0 1.0 1.0 >> $GAUSS
# python poisson_solver.py -n 11 21 41 81 161 -m sor -a 1.92 1.92 1.92 1.92 1.92 >> $SOR
python poisson_solver.py -n 11 21 41 81 161 -m multi-grid -a 1.0 1.0 1.0 1.0 1.0 >> $MG

python plot_csv.py -i $JACOBI $GAUSS $SOR -x 0 -y 3 -t "Computation Time vs. Grid Size" -o "$RESULTS_DIR/times_ref_slopes.png" -L -l -p 4.3 4.3 3.0
python plot_csv.py -i $JACOBI $GAUSS $SOR $MG -x 0 -y 3 -t "Computation Time vs. Grid Size" -o "$RESULTS_DIR/times_no_slopes.png" -L -l -p NaN NaN NaN NaN
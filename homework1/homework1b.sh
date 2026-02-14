#!/bin/bash

OUTFILE=results/homework1b_results.csv
rm -f $OUTFILE
echo "N, Error (MAE), Time (seconds)" >> $OUTFILE
python scalar_transport.py -b >> $OUTFILE

python plot_csv.py -x 0 -y 1 -n 100 -f $OUTFILE -t "Mean Absolute Error vs. Grid Size" -o "results/homework1b_errors.png" -l
python plot_csv.py -x 0 -y 2 -n 100 -f $OUTFILE -t "Calculation Time vs. Grid Size" -o "results/homework1b_times.png"
#!/bin/bash

OUTFILE=results/homework1c_results.csv
rm -f $OUTFILE
echo "Q1, Q0, Error (MAE), Time (seconds), Iterations" >> $OUTFILE
python scalar_transport.py -c >> $OUTFILE

# python plot_csv.py -x 0 -y 2 -f $OUTFILE -t "Mean Absolute Error vs. Grid Size" -o "results/homework1c_errors.png" -l
# python plot_csv.py -x 0 -y 3 -f $OUTFILE -t "Calculation Time vs. Grid Size" -o "results/homework1c_times.png"
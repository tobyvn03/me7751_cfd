#!/bin/bash

python solver.py -i input_part3b.csv -o output_part3b.csv -f part3b -p plot_part3b -v --plot_every 10000

python plot_csv.py -i part3b/output_part3b.csv -x -3 -y -2 -o "part3b/plot_part3b_spatial_conv.png" -L -l -p -14
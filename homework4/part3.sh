#!/bin/bash

python channel_flow.py -i input_part3.csv -o output_part3.csv -f part3 -p plot_part3

python plot_csv.py -i part3/output_part3.csv -x 3 -y -2 -o "part3/plot_part3_errors.png" -p nan
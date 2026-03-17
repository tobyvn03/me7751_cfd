#!/bin/bash

python channel_flow.py -i input_part7.csv -o output_part7.csv -f part7 -p plot_part7

python plot_csv.py -i part7/output_part7.csv -x 6 -y -3 -o "part7/plot_part7_entrance_len.png" -p nan
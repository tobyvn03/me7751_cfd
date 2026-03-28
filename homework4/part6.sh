#!/bin/bash

python channel_flow.py -i input_part6.csv -o output_part6.csv -f part6 -p plot_part6

python plot_csv.py -i part6/output_part6.csv -x 11 -y -2 -o "part6/plot_part6_spatial_conv.png" -L -l -p 2.2
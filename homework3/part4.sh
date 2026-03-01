#!/bin/bash

python scalar_transport.py -i input_spatial_convergence_crank-nicolson.csv -o spatial_convergence_crank-nicolson.csv -f part4

python plot_csv.py -i part4/spatial_convergence_crank-nicolson.csv -x 8 -y -2 -t "Spatial Convergence" -o "part4/plots/spatial_convergence_crank-nicolson.png" -L -l -p 2.0
#!/bin/bash

# python scalar_transport.py -i input_temporal_convergence_explicit.csv -o temporal_convergence_explicit.csv -f part5
# python scalar_transport.py -i input_temporal_convergence_implicit.csv -o temporal_convergence_implicit.csv -f part5
# python scalar_transport.py -i input_temporal_convergence_crank-nicolson.csv -o temporal_convergence_crank-nicolson.csv -f part5

python plot_csv.py -i part5/temporal_convergence_explicit.csv part5/temporal_convergence_implicit.csv part5/temporal_convergence_crank-nicolson.csv -x 6 -y -2 -t "Temporal Convergence" -o "part5/plots/temporal_convergence.png" -L -l -p 0.7 0.6 0.007
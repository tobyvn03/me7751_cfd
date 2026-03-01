#!/bin/bash

# python scalar_transport.py -i input_computational_cost_explicit.csv -o computational_cost_explicit.csv -f part7
# python scalar_transport.py -i input_computational_cost_implicit.csv -o computational_cost_implicit.csv -f part7
# python scalar_transport.py -i input_computational_cost_crank-nicolson.csv -o computational_cost_crank-nicolson.csv -f part7

python plot_csv.py -i part7/computational_cost_explicit.csv part7/computational_cost_implicit.csv part7/computational_cost_crank-nicolson.csv -x 1 -y -1 -t "Computational Cost" -o "part7/plots/computational_cost.png" -L -l -p 1.0 1.0 1.0
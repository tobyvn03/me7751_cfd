#!/bin/bash

python scalar_transport.py -i input_explicit_case_1.csv -o explicit_case_1.csv -f part2
python scalar_transport.py -i input_implicit_case_1.csv -o implicit_case_1.csv -f part2
python scalar_transport.py -i input_crank-nicolson_case_1.csv -o crank-nicolson_case_1.csv -f part2
python scalar_transport.py -i input_explicit_case_2.csv -o explicit_case_2.csv -f part2
python scalar_transport.py -i input_implicit_case_2.csv -o implicit_case_2.csv -f part2
python scalar_transport.py -i input_crank-nicolson_case_2.csv -o crank-nicolson_case_2.csv -f part2
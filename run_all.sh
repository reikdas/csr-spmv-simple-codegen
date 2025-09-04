#!/bin/bash
taskset -a -c $1 python3 create_csr.py
taskset -a -c $1 python3 eval.py
taskset -a -c $1 python3 plot_nums.py
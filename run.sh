#!/bin/bash
./cleanup.sh
taskset -a -c 0 python3 eval_spv8.py
#!/bin/bash
./download.sh
python3 create_csr.py
python3 spv8-public/contrib/generate_mtx.py
taskset -a -c 0 python3 eval_spv8.py
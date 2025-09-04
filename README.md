Put all `.mtx` files you want to operate over in `matrices/`.

Run in order:

* python3 create_csr.py # Pass any mtx file of your choice
* python3 eval.py # Generate code + Evaluate
* python3 plot_nums.py # Create plot from collected times

Or, look at `run_all.sh` which runs all of these.
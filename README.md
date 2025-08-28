Run in order:

* python3 create_csr.py # Pass any mtx file of your choice
* python3 reduce_csr.py # Create reduced nnz versions
* python3 mtx_to_csr_spmv.py # Generate code + Evaluate
* python3 plot_nums.py # Create plot from collected times

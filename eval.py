import csv
import glob
import os
import subprocess
import shutil
import sys
from pathlib import Path

CFLAGS = ["-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-ffast-math", "-I/home/min/a/das160/papi-install/include"]

# Global variable to store timing results
timing_results = []

def write_dense_vector(val: float, size: int):
    """Inline version of write_dense_vector function."""
    filename = f"generated_vector_{size}.vector"
    dir_name = "Generated_dense_tensors"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename), "w") as f:
        x = [val] * size
        f.write(f"{','.join(map(str, x))}\n")

def write_dense_matrix(val: float, m: int, n: int):
    filename = f"generated_matrix_{m}x{n}.matrix"
    dir_name = "Generated_dense_tensors"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename), "w") as f:
        x = [val] * n * m
        f.write(f"{','.join(map(str, x))}\n")

def read_csr_file(filepath):
    """Read a .csr file and return the matrix dimensions and nnz."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse indptr line
        indptr_line = lines[0].strip()
        indptr_str = indptr_line.replace("indptr=[", "").replace("]", "")
        indptr = [int(x) for x in indptr_str.split(",")]
        
        # Parse indices line
        indices_line = lines[1].strip()
        indices_str = indices_line.replace("indices=[", "").replace("]", "")
        indices = [int(x) for x in indices_str.split(",")]
        
        # Parse data line
        data_line = lines[2].strip()
        data_str = data_line.replace("data=[", "").replace("]", "")
        data = [float(x) for x in data_str.split(",")]
        
        # Calculate dimensions
        rows = len(indptr) - 1
        cols = max(indices) + 1 if indices else 0
        nnz = len(data)
        
        print(f"Successfully read CSR file: {filepath}")
        print(f"Matrix shape: {rows} x {cols}")
        print(f"Number of non-zeros: {nnz}")
        
        return rows, cols, nnz
        
    except Exception as e:
        print(f"Error reading .csr file {filepath}: {e}")
        return None, None, None

def compile_c_program(c_filename, executable_name="spmv"):
    """Compile the C program using the flags from consts.py."""
    try:
        compile_cmd = ["gcc"] + CFLAGS + ["-o", executable_name, c_filename] + ["-L/home/min/a/das160/papi-install/lib", "-lpapi"]
        
        print(f"Compiling C program...")
        print(f"Command: {' '.join(compile_cmd)}")
        
        subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        
        print(f"✓ Compilation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed:")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Error: gcc compiler not found")
        return False

def execute_program(executable_name, mtx_path):
    """Execute the compiled SpMV program and extract timing information."""
    try:
        print(f"\nExecuting program...")
        print(f"Command: {executable_name} {mtx_path}")
        # Resolve the executable path and ensure it's executable.
        resolved_exec = executable_name
        # If a simple name was provided, try to find it on PATH
        if not os.path.isabs(resolved_exec):
            which_res = shutil.which(resolved_exec)
            if which_res:
                resolved_exec = which_res

        if not os.path.exists(resolved_exec) or not os.access(resolved_exec, os.X_OK):
            # Raise FileNotFoundError to be handled below with a friendly message
            raise FileNotFoundError(f"Executable not found or not executable: {resolved_exec}")

        # Call subprocess with a list of arguments (no shell). This avoids trying to
        # execute a single string containing both the program and its argument.
        result = subprocess.run([resolved_exec, mtx_path], capture_output=True, text=True, check=True)
        
        print(f"✓ Execution successful!")
        
        # Extract timing information from output
        timing_info = extract_timing(result.stdout)
        if timing_info is not None:
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"{timing_info:.6f}")
            print("=" * 60)
            return timing_info
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Execution failed:")
        print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Standard output: {e.stdout}")
        return None
    except FileNotFoundError:
        print(f"✗ Error: Executable {executable_name} not found")
        return None

def extract_timing(output_text):
    """Extract timing information from the program output."""
    try:
        # Look for the median timing line in the output
        for line in output_text.split('\n'):
            if "Time:" in line:
                # Extract the time value
                time_str = line.split(":")[-1].strip().split()[0]
                return float(time_str)
        return None
    except (ValueError, IndexError):
        return None

def csr_operation(csr_filepath, operation_type, bench_freq, result_type):
    if operation_type != "SpMV":
        raise Exception(f"Invalid operation type: {operation_type}")

    if result_type == "br_mispreds":
        executable_name = "/local/scratch/a/das160/csr-spmv-simple-codegen/CSR5_avx2/spmv_branch"
    elif result_type == "timing":
        executable_name = "/local/scratch/a/das160/csr-spmv-simple-codegen/CSR5_avx2/spmv_time"
    else:
        raise Exception(f"Invalid result type: {result_type}")
    
    print(f"\n{'='*80}")
    print(f"Processing: {csr_filepath}")
    print(f"{'='*80}")
    return execute_program(executable_name, csr_filepath)

def run_sparse_operation(matrix, operation_type, reduction_type, bench_freq, result_type):
    if result_type == "br_mispreds":
        f_name = "papi"
        col_name = "branch_mispreds"
    elif result_type == "timing":
        f_name = "timing"
        col_name = "time"
    else:
        raise Exception(f"Invalid result type: {result_type}")
    
    timing_results = {}
    
    # Process original matrix (100%)
    timing_results[100] = csr_operation(f"matrices/{matrix}.mtx", operation_type, bench_freq, result_type)
    
    # Process reduced CSR files
    csr_files = glob.glob(f"csr_files/{matrix}_{reduction_type}_*pct.mtx")
    for csr_file in csr_files:
        reduction_pct = csr_file.split(f"{reduction_type}_")[1].split("pct.mtx")[0]
        percentage = 100 - int(reduction_pct)
        timing_results[percentage] = csr_operation(csr_file, operation_type, bench_freq, result_type)

    # Write results to CSV
    with open(f"results/{f_name}_{matrix}_{operation_type}_{reduction_type}.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Percentage', col_name])
        
        # Sort results by percentage (descending)
        sorted_results = sorted(timing_results.items(), key=lambda x: x[0], reverse=True)
        # print(sorted_results)
        
        for percentage, time in sorted_results:
            if time is not False and time is not None:
                # print(percentage, time)
                writer.writerow([percentage, f"{time:.6f}"])

#!/usr/bin/env python3

import sys
import os
import subprocess
from scipy.io import mmread
import glob
import csv

CFLAGS = ["-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-ffast-math"]

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

def read_mtx_file(filepath):
    """Read a .mtx file and return the matrix."""
    try:
        matrix = mmread(filepath)
        print(f"Successfully read matrix from {filepath}")
        print(f"Matrix shape: {matrix.shape}")
        
        # Handle both sparse and dense matrices
        if hasattr(matrix, 'nnz'):
            print(f"Number of non-zeros: {matrix.nnz}")
        return matrix
    except Exception as e:
        print(f"Error reading .mtx file: {e}")
        sys.exit(1)

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
        compile_cmd = ["gcc"] + CFLAGS + ["-o", executable_name, c_filename]
        
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

def execute_spmv_program(executable_name="spmv"):
    """Execute the compiled SpMV program and extract timing information."""
    try:
        print(f"\nExecuting SpMV program...")
        print(f"Command: ./{executable_name}")
        
        result = subprocess.run([f"./{executable_name}"], capture_output=True, text=True, check=True)
        
        print(f"✓ Execution successful!")
        
        # Extract timing information from output
        timing_info = extract_timing(result.stdout)
        if timing_info:
            print("\n" + "=" * 60)
            print("TIMING RESULTS")
            print("=" * 60)
            print(f"Time: {timing_info:.6f} ns")
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
    
def generate_c_program(csr_filename, vector_filename, rows, cols, nnz, output_filename):
    c_code = f"""
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void spmv_sparse(double *restrict y, const double *restrict csr_val, const int *restrict indices, const int *restrict indptr, const double *restrict x, const int rpntr_size) {{
	double sum = 0;
    for (int i = 0; i < rpntr_size; i++) {{
        sum = 0;
		for (int j = indptr[i]; j < indptr[i+1]; j++) {{
			sum += csr_val[j] * x[indices[j]];
		}}
        y[i] += sum;
	}}
}}

int main() {{
    double *y = (double*)malloc({rows} * sizeof(double));
    double *x = (double*)malloc({cols} * sizeof(double));
    double *csr_val = (double*)malloc({nnz} * sizeof(double));
    int *indices = (int*)malloc({nnz} * sizeof(int));
    int *indptr = (int*)malloc(({rows} + 1) * sizeof(int));
    FILE *file1 = fopen("{csr_filename}", "r");
    if (file1 == NULL) {{
        perror("Error opening file1");
        exit(EXIT_FAILURE);
    }}
    FILE *file2 = fopen("Generated_dense_tensors/{vector_filename}", "r");
    if (file2 == NULL) {{
        perror("Error opening file2");
        exit(EXIT_FAILURE);
    }}
    memset(x, 0, sizeof(double)*{cols});
    memset(csr_val, 0, sizeof(double)*{nnz});
    memset(indices, 0, sizeof(int)*{nnz});
    memset(indptr, 0, sizeof(int)*({rows} + 1));
    char c;
    int x_size=0, val_size=0;
    assert(fscanf(file1, "indptr=[%c", &c) == 1);
    if (c != ']') {{
        ungetc(c, file1);
        assert(fscanf(file1, "%d", &indptr[val_size]) == 1);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
    }}
    assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');
    val_size=0;
    assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
    val_size++;
    while (1) {{
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {{
            assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
            val_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    if(fscanf(file1, "%c", &c));
    assert(c=='\\n');
    val_size=0;
    assert(fscanf(file1, "data=[%lf", &csr_val[val_size]) == 1.0);
    val_size++;
    while (1) {{
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {{
            assert(fscanf(file1, "%lf", &csr_val[val_size]) == 1.0);
            val_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    fclose(file1);
    while (x_size < {cols} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);
    struct timespec t1, t2;
    double times[100];
    for (int i=0; i<100; i++) {{
        memset(y, 0, sizeof(double)*{rows});
        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmv_sparse(y, csr_val, indices, indptr, x, {rows});
        clock_gettime(CLOCK_MONOTONIC, &t2);
        times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }}
    for (int i=0; i<99; i++) {{
        for (int j=i+1; j<100; j++) {{
            if (times[j] < times[i]) {{
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }}
        }}
    }}
    printf("Time: %.2f ns\\n", times[50]);
    for (int i=0; i<{rows}; i++) {{
        printf("%.2f\\n", y[i]);
    }}
    free(y);
    free(x);
    free(csr_val);
    free(indptr);
    free(indices);
}}"""
    try:
        with open(output_filename, 'w') as f:
            f.write(c_code)
        print(f"C program generated and saved to {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error generating C program: {e}")
        sys.exit(1)

def process_csr_file(csr_filepath):
    """Process a single CSR file and run SpMV evaluation."""
    print(f"\n{'='*80}")
    print(f"Processing: {csr_filepath}")
    print(f"{'='*80}")
    
    # Extract percentage from filename
    filename = os.path.basename(csr_filepath)
    percentage = 100  # Default for foo.csr
    
    if "reduced_" in filename and "pct.csr" in filename:
        reduction_pct = filename.split("reduced_")[1].split("pct.csr")[0]
        percentage = 100 - int(reduction_pct)  # Convert reduction % to retention %
        print(f"Matrix reduction: {reduction_pct}% → Retention: {percentage}%")
    elif filename == "foo.csr":
        print(f"Original matrix: 100%")
    else:
        print(f"Unknown filename format: {filename}")
        return False
    
    # Read CSR file to get dimensions
    rows, cols, nnz = read_csr_file(csr_filepath)
    if rows is None or cols is None or nnz is None:
        print(f"Skipping {csr_filepath} due to read error")
        return False
    
    # Generate vector for this matrix size
    write_dense_vector(1.0, cols)
    
    # Generate C program
    c_filename = generate_c_program(
        csr_filepath,
        f"generated_vector_{cols}.vector",
        rows=rows,
        cols=cols,
        nnz=nnz,
        output_filename="spmv.c"
    )
    
    # Compile C program
    if c_filename and compile_c_program(c_filename, "spmv"):
        # Execute SpMV program and get timing
        timing = execute_spmv_program("spmv")
        if timing is not None:
            # Store result for CSV output
            timing_results.append((percentage, timing))
            return True
        else:
            print(f"Failed to get timing for {csr_filepath}")
            return False
    else:
        print(f"Skipping execution for {csr_filepath} due to compilation failure.")
        return False

def write_timing_results_to_csv(output_filename="timing_results.csv"):
    """Write timing results to a CSV file."""
    try:
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Percentage', 'Time_ns'])
            
            # Sort results by percentage (descending)
            sorted_results = sorted(timing_results, key=lambda x: int(x[0]), reverse=True)
            
            for percentage, time in sorted_results:
                writer.writerow([percentage, f"{time:.6f}"])
        
        print(f"\n✓ Timing results written to {output_filename}")
        print(f"Results:")
        for percentage, time in sorted_results:
            print(f"  {percentage}%: {time:.6f} ns")
            
    except Exception as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    # Clear previous results
    timing_results = []
    
    # Process foo.csr first (100%)
    if os.path.exists("foo.csr"):
        process_csr_file("foo.csr")
    
    # Process reduced CSR files
    csr_files = glob.glob("foo_reduced_*pct.csr")
    
    if not csr_files:
        print("No foo_reduced_*pct.csr files found!")
        if not timing_results:
            print("No timing results to write!")
            sys.exit(1)
    else:
        # Process each reduced CSR file
        for csr_file in csr_files:
            process_csr_file(csr_file)
    
    # Write results to CSV
    if timing_results:
        write_timing_results_to_csv()
    else:
        print("No timing results collected!")
        sys.exit(1)
 
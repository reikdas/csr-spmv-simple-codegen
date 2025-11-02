#!/usr/bin/env python3
"""
Simple Row Reordering SpMV Evaluation
Just reorders rows by column locality - no column blocking overhead.
This is the minimal overhead approach to test if locality helps.
"""

import csv
import glob
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

CFLAGS = ["-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-ffast-math"]

def write_dense_vector(val: float, size: int):
    """Generate dense vector for SpMV."""
    filename = f"generated_vector_{size}.vector"
    dir_name = "Generated_dense_tensors"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename), "w") as f:
        x = [val] * size
        f.write(f"{','.join(map(str, x))}\n")

def read_csr_file(filepath):
    """Read a .csr file and return the matrix dimensions, nnz, and data."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        indptr_line = lines[0].strip()
        indptr_str = indptr_line.replace("indptr=[", "").replace("]", "")
        indptr = [int(x) for x in indptr_str.split(",")]
        
        indices_line = lines[1].strip()
        indices_str = indices_line.replace("indices=[", "").replace("]", "")
        indices = [int(x) for x in indices_str.split(",")]
        
        data_line = lines[2].strip()
        data_str = data_line.replace("data=[", "").replace("]", "")
        data = [float(x) for x in data_str.split(",")]
        
        rows = len(indptr) - 1
        cols = max(indices) + 1 if indices else 0
        nnz = len(data)
        
        return rows, cols, nnz, indptr, indices, data
        
    except Exception as e:
        print(f"Error reading .csr file {filepath}: {e}")
        return None, None, None, None, None, None

def compute_row_locality_key(indptr, indices, row_idx):
    """
    Compute a locality key for a row.
    We use min_col + max_col to group rows by column range.
    This is simpler and faster than median.
    """
    start_idx = indptr[row_idx]
    end_idx = indptr[row_idx+1]
    
    if start_idx >= end_idx:
        return 0  # Empty row
    
    cols = indices[start_idx:end_idx]
    min_col = int(np.min(cols))
    max_col = int(np.max(cols))
    
    # Use min + max as locality key
    # Rows with similar column ranges will be grouped
    return min_col + max_col

def create_row_ordering(indptr, indices, rows):
    """
    Create row ordering to improve column locality.
    Returns array mapping new_row_idx -> old_row_idx.
    """
    row_keys = []
    
    for i in range(rows):
        key = compute_row_locality_key(indptr, indices, i)
        row_keys.append((key, i))
    
    # Sort by locality key
    row_keys.sort(key=lambda x: x[0])
    
    # Extract reordered row indices
    row_order = [old_idx for _, old_idx in row_keys]
    
    return row_order

def compile_c_program(c_filename, executable_name="spmv"):
    """Compile the C program."""
    try:
        compile_cmd = ["gcc"] + CFLAGS + ["-o", executable_name, c_filename]
        subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Error: gcc compiler not found")
        return False

def execute_program(executable_name="spmv"):
    """Execute the compiled SpMV program and extract timing information."""
    try:
        result = subprocess.run([f"./{executable_name}"], capture_output=True, text=True, check=True)
        
        # Extract timing information from output
        for line in result.stdout.split('\n'):
            if "Time:" in line:
                time_str = line.split(":")[-1].strip().split()[0]
                return float(time_str)
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Execution failed: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"✗ Error: Executable {executable_name} not found")
        return None

def generate_reordered_spmv_timing(csr_filename, vector_filename, rows, cols, nnz, 
                                   indptr, indices, data, output_filename, bench_freq):
    """
    Generate SpMV with simple row reordering - no column blocking.
    
    Key idea: Process rows in order of column locality, but each row is still
    processed completely in one pass. Minimal overhead.
    """
    
    # Create row ordering
    row_order = create_row_ordering(indptr, indices, rows)
    row_order_str = ','.join(map(str, row_order))
    
    c_code = f"""
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Simple reordered SpMV - just changes row processing order
void spmv_reordered(double *restrict y, const double *restrict csr_val, 
                    const int *restrict indices, const int *restrict indptr, 
                    const double *restrict x, const int *restrict row_order, int rows) {{
    // Process rows in reordered sequence for better column locality
    for (int new_i = 0; new_i < rows; new_i++) {{
        int i = row_order[new_i];  // Get original row index
        double sum = 0.0;
        
        // Process this row completely (no column blocking overhead)
        for (int j = indptr[i]; j < indptr[i+1]; j++) {{
            sum += csr_val[j] * x[indices[j]];
        }}
        
        y[i] = sum;  // Write to original position (no permutation needed!)
    }}
}}

int main() {{
    double *y = (double*)malloc({rows} * sizeof(double));
    double *x = (double*)malloc({cols} * sizeof(double));
    double *csr_val = (double*)malloc({nnz} * sizeof(double));
    int *indices = (int*)malloc({nnz} * sizeof(int));
    int *indptr = (int*)malloc(({rows} + 1) * sizeof(int));
    
    // Row ordering (compile-time constant)
    int row_order[] = {{{row_order_str}}};
    
    struct timespec t1, t2;
    double times[{bench_freq}];
    
    for (int iter = 0; iter < {bench_freq}; iter++) {{
        FILE *file1 = fopen("{csr_filename}", "r");
        if (file1 == NULL) {{
            perror("Error opening CSR file");
            exit(EXIT_FAILURE);
        }}
        FILE *file2 = fopen("Generated_dense_tensors/{vector_filename}", "r");
        if (file2 == NULL) {{
            perror("Error opening vector file");
            exit(EXIT_FAILURE);
        }}
        
        memset(x, 0, sizeof(double)*{cols});
        memset(csr_val, 0, sizeof(double)*{nnz});
        memset(indices, 0, sizeof(int)*{nnz});
        memset(indptr, 0, sizeof(int)*({rows} + 1));
        
        char c;
        int x_size=0, val_size=0;
        
        // Parse indptr
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
        
        // Parse indices
        val_size=0;
        assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indices[val_size]) == 1);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');
        
        // Parse data
        val_size=0;
        assert(fscanf(file1, "data=[%lf", &csr_val[val_size]) == 1);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%lf", &csr_val[val_size]) == 1);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        fclose(file1);
        
        // Load vector
        while (x_size < {cols} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);
        
        memset(y, 0, sizeof(double)*{rows});
        
        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmv_reordered(y, csr_val, indices, indptr, x, row_order, {rows});
        clock_gettime(CLOCK_MONOTONIC, &t2);
        
        times[iter] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }}
    
    // Sort and take median
    for (int i = 0; i < {bench_freq - 1}; i++) {{
        for (int j = i + 1; j < {bench_freq}; j++) {{
            if (times[j] < times[i]) {{
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }}
        }}
    }}
    
    printf("Time: %.2f ns\\n", times[{bench_freq // 2}]);
    
    for (int i = 0; i < {rows}; i++) {{
        printf("%.2f\\n", y[i]);
    }}
    
    free(y);
    free(x);
    free(csr_val);
    free(indptr);
    free(indices);
    return 0;
}}
"""
    
    try:
        with open(output_filename, 'w') as f:
            f.write(c_code)
        return output_filename
    except Exception as e:
        print(f"Error generating C program: {e}")
        sys.exit(1)

def csr_operation(csr_filepath, bench_freq):
    """Process a single CSR file."""
    
    rows, cols, nnz, indptr, indices, data = read_csr_file(csr_filepath)
    if rows is None:
        return None
    
    write_dense_vector(1.0, cols)
    vector_filename = f"generated_vector_{cols}.vector"
    
    c_filename = generate_reordered_spmv_timing(
        csr_filepath,
        vector_filename,
        rows=rows,
        cols=cols,
        nnz=nnz,
        indptr=indptr,
        indices=indices,
        data=data,
        output_filename="spmv_reordered.c",
        bench_freq=bench_freq
    )
    
    if c_filename and compile_c_program(c_filename, "spmv_reordered"):
        return execute_program("spmv_reordered")
    
    return None

def run_sparse_operation(matrix, operation_type, reduction_type, bench_freq):
    """Run reordered SpMV on matrix with different reduction percentages."""
    
    timing_results = {}
    
    # Process original matrix (100%)
    print(f"  Processing 100%...")
    timing_results[100] = csr_operation(f"csr_files/{matrix}.csr", bench_freq)
    
    # Process reduced CSR files
    csr_files = glob.glob(f"csr_files/{matrix}_{reduction_type}_*pct.csr")
    for csr_file in sorted(csr_files):
        reduction_pct = csr_file.split(f"{reduction_type}_")[1].split("pct.csr")[0]
        percentage = 100 - int(reduction_pct)
        print(f"  Processing {percentage}%...")
        timing_results[percentage] = csr_operation(csr_file, bench_freq)

    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Write results to CSV with _reordered suffix
    output_file = f"results/timing_{matrix}_{operation_type}_{reduction_type}_reordered.csv"
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Percentage', 'time'])
        
        sorted_results = sorted(timing_results.items(), key=lambda x: x[0], reverse=True)
        
        for percentage, time in sorted_results:
            if time is not False and time is not None:
                writer.writerow([percentage, f"{time:.6f}"])
    
    print(f"✓ Saved: {output_file}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING SIMPLE REORDERED SpMV (minimal overhead)")
    print("="*80 + "\n")
    
    matrices = [p.stem for p in Path("matrices").glob("*.mtx")]
    reduction_types = ["random"]
    
    for matrix in matrices:
        print(f"\nProcessing: {matrix}")
        for reduction_type in reduction_types:
            run_sparse_operation(matrix, "spmv", reduction_type, 100)
    
    print("\n" + "="*80)
    print("✓ SIMPLE REORDERED version complete!")
    print("="*80)
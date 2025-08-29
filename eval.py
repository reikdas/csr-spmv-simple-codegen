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

def execute_program(executable_name="spmv"):
    """Execute the compiled SpMV program and extract timing information."""
    try:
        print(f"\nExecuting program...")
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
    
def generate_spmm(csr_filename, matrix_filename, sparse_rows, sparse_cols, dense_cols, nnz, output_filename):
    c_code = f"""
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void spmm_sparse(double *restrict y, const double *restrict csr_val, const int *restrict indices, const int *restrict indptr, const double *restrict x, const int sparse_rows, const int dense_cols) {{
    for (int i = 0; i < sparse_rows; i++) {{
        for (int p = indptr[i]; p < indptr[i+1]; p++) {{
            int col = indices[p];
            double val = csr_val[p];
            for (int j = 0; j < dense_cols; ++j) {{
                y[i * dense_cols + j] += val * x[col * dense_cols + j];
            }}
        }}
    }}
}}

int main() {{
    double *y = (double*)malloc({sparse_rows*dense_cols}*sizeof(double));
    double *x = (double*)malloc({sparse_cols*dense_cols}*sizeof(double));
    double *csr_val = (double*)malloc({nnz}*sizeof(double));
    int *indices = (int*)malloc({nnz}*sizeof(int));
    int *indptr = (int*)malloc(({sparse_rows + 1}) * sizeof(int));
    FILE *file1 = fopen("{csr_filename}", "r");
    if (file1 == NULL) {{
        perror("Error opening file1\\n");
        exit(EXIT_FAILURE);
    }}
    FILE *file2 = fopen("Generated_dense_tensors/{matrix_filename}", "r");
    if (file2 == NULL) {{
        perror("Error opening file2\\n");
        exit(EXIT_FAILURE);
    }}
    memset(x, 0, sizeof(double)*{sparse_cols*dense_cols});
    memset(csr_val, 0, sizeof(double)*{nnz});
    memset(indices, 0, sizeof(int)*{nnz});
    memset(indptr, 0, sizeof(int)*({sparse_rows} + 1));
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
    while (x_size < {sparse_cols*dense_cols} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);
    struct timespec t1, t2;
    double times[100];
    for (int i = 0; i < 100; i++) {{
        memset(y, 0, sizeof(double)*{sparse_rows*dense_cols});
        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmm_sparse(y, csr_val, indices, indptr, x, {sparse_rows}, {dense_cols});
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
    for (int i=0; i<{sparse_rows * dense_cols}; i++) {{
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


def generate_spmv(csr_filename, vector_filename, rows, cols, nnz, output_filename):
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
        y[i] = sum;
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

def csr_spmm(csr_filepath):
    """Process a single CSR file and run SpMM evaluation."""
    print(f"\n{'='*80}")
    print(f"Processing: {csr_filepath}")
    print(f"{'='*80}")

    rows, cols, nnz = read_csr_file(csr_filepath)

    write_dense_matrix(1.0, cols, 512)

    c_filename = generate_spmm(
        csr_filepath,
        f"generated_matrix_{cols}x{512}.matrix",
        rows,
        cols, 512,
        nnz=nnz,
        output_filename="spmm.c"
    )

    if compile_c_program(c_filename, "spmm"):
        return execute_program("spmm")
    else:
        print(f"Skipping execution for {csr_filepath}_spmm due to compilation failure.")
        return False

def csr_spmv(csr_filepath):
    """Process a single CSR file and run SpMV evaluation."""
    print(f"\n{'='*80}")
    print(f"Processing: {csr_filepath}")
    print(f"{'='*80}")
    
    # Read CSR file to get dimensions
    rows, cols, nnz = read_csr_file(csr_filepath)
    
    # Generate vector for this matrix size
    write_dense_vector(1.0, cols)
    
    # Generate C program
    c_filename = generate_spmv(
        csr_filepath,
        f"generated_vector_{cols}.vector",
        rows=rows,
        cols=cols,
        nnz=nnz,
        output_filename="spmv.c"
    )
    
    if c_filename and compile_c_program(c_filename, "spmv"):
        return execute_program("spmv")    
    else:
        print(f"Skipping execution for {csr_filepath}_spmv due to compilation failure.")
        return False


def run_spmm(matrix):
    timing_results = {}
    
    timing_results[100] = csr_spmm(f"csr_files/{matrix}.csr")
    
    # Process reduced CSR files
    csr_files = glob.glob(f"csr_files/{matrix}_reduced_*pct.csr")
    for csr_file in csr_files:
        reduction_pct = csr_file.split("reduced_")[1].split("pct.csr")[0]
        percentage = 100 - int(reduction_pct)
        timing_results[percentage] = csr_spmm(csr_file)

    with open(f"timing_{matrix}_spmm.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Percentage', 'Time_ns'])
        
        # Sort results by percentage (descending)
        sorted_results = sorted(timing_results.items(), key=lambda x: x[0], reverse=True)
        
        for percentage, time in sorted_results:
            if time is not False:
                writer.writerow([percentage, f"{time:.6f}"])

def run_spmv(matrix):
    timing_results = {}
    
    timing_results[100] = csr_spmv(f"csr_files/{matrix}.csr")
    
    # Process reduced CSR files
    csr_files = glob.glob(f"csr_files/{matrix}_reduced_*pct.csr")
    for csr_file in csr_files:
        reduction_pct = csr_file.split("reduced_")[1].split("pct.csr")[0]
        percentage = 100 - int(reduction_pct)
        timing_results[percentage] = csr_spmv(csr_file)

    with open(f"timing_{matrix}_spmv.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Percentage', 'Time_ns'])
        
        # Sort results by percentage (descending)
        sorted_results = sorted(timing_results.items(), key=lambda x: x[0], reverse=True)
        
        for percentage, time in sorted_results:
            if time is not False:
                writer.writerow([percentage, f"{time:.6f}"])


if __name__ == "__main__":
    # matrices = ["brainpc2", "heart1", "lowThrust_7"]
    matrices = ["brainpc2"]
    for matrix in matrices:
        # run_spmv(matrix)
        run_spmm(matrix)

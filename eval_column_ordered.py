#!/usr/bin/env python3
"""
Fixed Column-Blocked SpMV Evaluation
Pre-partitions CSR data by column blocks to avoid redundant iteration.
This is the correct way to do column blocking efficiently.
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

def partition_csr_by_column_blocks(indptr, indices, data, rows, cols, col_block_size=512):
    """
    Pre-partition CSR data by column blocks.

    Returns: List of (block_id, row_segments) where each row_segment is:
             (row_id, col_indices, values)

    Key insight: We separate the CSR data so each row segment only contains
    nonzeros in one column block. This avoids redundant iteration.
    """
    num_col_blocks = (cols + col_block_size - 1) // col_block_size

    # For each column block, store row segments
    # block_segments[block_id] = [(row_id, [indices], [values]), ...]
    block_segments = [[] for _ in range(num_col_blocks)]

    for row in range(rows):
        start_idx = indptr[row]
        end_idx = indptr[row+1]

        if start_idx >= end_idx:
            continue  # Empty row

        # Group this row's nonzeros by which column block they belong to
        row_by_blocks = {}

        for j in range(start_idx, end_idx):
            col = indices[j]
            val = data[j]
            block_id = min(col // col_block_size, num_col_blocks - 1)

            if block_id not in row_by_blocks:
                row_by_blocks[block_id] = ([], [])

            row_by_blocks[block_id][0].append(col)
            row_by_blocks[block_id][1].append(val)

        # Add row segments to appropriate blocks
        for block_id, (cols_in_block, vals_in_block) in row_by_blocks.items():
            block_segments[block_id].append((row, cols_in_block, vals_in_block))

    # Sort row segments within each block by row id for better cache behavior
    for block_id in range(num_col_blocks):
        block_segments[block_id].sort(key=lambda x: x[0])

    return block_segments

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

def generate_fixed_colblocked_spmv_timing(csr_filename, vector_filename, rows, cols, nnz,
                                          indptr, indices, data, output_filename, bench_freq,
                                          col_block_size=512):
    """
    Generate efficient column-blocked SpMV with pre-partitioned data.

    Key difference from broken version: We pre-partition the CSR data at
    code generation time, so we never iterate over irrelevant nonzeros.
    """

    # Pre-partition the data
    block_segments = partition_csr_by_column_blocks(indptr, indices, data, rows, cols, col_block_size)

    # Generate code for each block
    num_col_blocks = len(block_segments)

    # Count total operations for verification
    total_ops = sum(len(seg[1]) for block in block_segments for seg in block)
    if total_ops != nnz:
        print(f"Warning: Partitioning inconsistency - {total_ops} ops vs {nnz} nnz")

    # Generate block processing code
    block_code_parts = []

    for block_id, segments in enumerate(block_segments):
        if not segments:
            continue

        col_start = block_id * col_block_size
        col_end = min((block_id + 1) * col_block_size, cols)

        # Generate code for this block
        block_code = f"""
    // Column block {block_id}: columns [{col_start}, {col_end}) - {len(segments)} row segments
    {{"""

        for row_id, cols_in_seg, vals_in_seg in segments:
            nnz_in_seg = len(cols_in_seg)

            # Embed this row segment's data
            cols_str = ','.join(map(str, cols_in_seg))
            vals_str = ','.join(map(str, vals_in_seg))

            block_code += f"""
        // Row {row_id}: {nnz_in_seg} nonzeros in this block
        {{
            int cols_{row_id}[] = {{{cols_str}}};
            double vals_{row_id}[] = {{{vals_str}}};
            double sum = 0.0;
            for (int k = 0; k < {nnz_in_seg}; k++) {{
                sum += vals_{row_id}[k] * x[cols_{row_id}[k]];
            }}
            y[{row_id}] += sum;
        }}"""

        block_code += """
    }"""

        block_code_parts.append(block_code)

    all_blocks_code = '\n'.join(block_code_parts)

    c_code = f"""
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Fixed column-blocked SpMV - pre-partitioned data
void spmv_colblocked_fixed(double *restrict y, const double *restrict x, int rows) {{
    // Process each column block with pre-partitioned segments
{all_blocks_code}
}}

int main() {{
    double *y = (double*)malloc({rows} * sizeof(double));
    double *x = (double*)malloc({cols} * sizeof(double));

    struct timespec t1, t2;
    double times[{bench_freq}];

    for (int iter = 0; iter < {bench_freq}; iter++) {{
        FILE *file2 = fopen("Generated_dense_tensors/{vector_filename}", "r");
        if (file2 == NULL) {{
            perror("Error opening vector file");
            exit(EXIT_FAILURE);
        }}

        memset(x, 0, sizeof(double)*{cols});

        int x_size = 0;
        while (x_size < {cols} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);

        memset(y, 0, sizeof(double)*{rows});

        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmv_colblocked_fixed(y, x, {rows});
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

    c_filename = generate_fixed_colblocked_spmv_timing(
        csr_filepath,
        vector_filename,
        rows=rows,
        cols=cols,
        nnz=nnz,
        indptr=indptr,
        indices=indices,
        data=data,
        output_filename="spmv_colblocked_fixed.c",
        bench_freq=bench_freq
    )

    if c_filename and compile_c_program(c_filename, "spmv_colblocked_fixed"):
        return execute_program("spmv_colblocked_fixed")

    return None

def run_sparse_operation(matrix, operation_type, reduction_type, bench_freq):
    """Run fixed column-blocked SpMV on matrix with different reduction percentages."""

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

    # Write results to CSV with _colblocked_fixed suffix
    output_file = f"results/timing_{matrix}_{operation_type}_{reduction_type}_colblocked_fixed.csv"
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
    print("RUNNING FIXED COLUMN-BLOCKED SpMV (pre-partitioned data)")
    print("="*80 + "\n")

    matrices = [p.stem for p in Path("matrices").glob("*.mtx")]
    reduction_types = ["random"]

    for matrix in matrices:
        print(f"\nProcessing: {matrix}")
        for reduction_type in reduction_types:
            run_sparse_operation(matrix, "spmv", reduction_type, 100)

    print("\n" + "="*80)
    print("✓ FIXED COLUMN-BLOCKED version complete!")
    print("="*80)
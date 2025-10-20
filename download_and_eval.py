import csv
import os
import tarfile

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse
from ssgetpy import search

from create_csr import create_csr_variants, parse_array
from eval import run_sparse_operation


def count_nnz_rows(mtx_path):
    # Read the matrix using scipy
    matrix = mmread(mtx_path)

    matrix_csr = matrix.tocsr()
    row_ptr = matrix_csr.indptr
        
    nnz_rows = 0
    
    for i in range(matrix_csr.shape[0]):
        if row_ptr[i+1] - row_ptr[i] > 0:
            nnz_rows += 1
            
    return nnz_rows

def is_scale_free(mtx_path: str) -> bool:
    """
    Check if a sparse matrix is scale-free.
    A matrix is considered scale-free if 90% of the non-zero elements
    are present in 10% of the rows.
    
    Args:
        mtx_path: Path to the matrix file
        
    Returns:
        bool: True if the matrix is scale-free, False otherwise
    """
    # Read the matrix using scipy
    matrix = mmread(mtx_path)
    
    # Convert to CSR format for efficient row-wise operations
    if issparse(matrix):
        matrix_csr = matrix.tocsr()  # type: ignore
    else:
        # If it's dense, convert to sparse first
        matrix_csr = csr_matrix(matrix)
    
    # Ensure we have a valid matrix
    if matrix_csr is None or matrix_csr.shape[0] == 0:  # type: ignore
        return False
    
    # Get the total number of non-zero elements
    total_nnz = matrix_csr.nnz
    if total_nnz == 0:
        return False
    
    # Calculate row-wise non-zero counts
    row_ptr = matrix_csr.indptr
    if row_ptr is None or len(row_ptr) == 0:
        return False
    
    # Type guard: ensure matrix_csr is not None
    assert matrix_csr is not None
        
    row_nnz_counts = []
    
    for i in range(matrix_csr.shape[0]):  # type: ignore
        if i + 1 < len(row_ptr):
            row_nnz = row_ptr[i+1] - row_ptr[i]
            row_nnz_counts.append(row_nnz)
        else:
            row_nnz_counts.append(0)
    
    # Sort rows by their non-zero count in descending order
    row_nnz_counts.sort(reverse=True)
    
    # Calculate thresholds
    total_rows = matrix_csr.shape[0]  # type: ignore
    target_rows = max(1, int(0.1 * total_rows))  # 10% of rows
    target_nnz = int(0.9 * total_nnz)  # 90% of nnz
    
    # Check if the top 10% of rows contain at least 90% of the nnz
    top_rows_nnz = sum(row_nnz_counts[:target_rows])
    
    return top_rows_nnz >= target_nnz

def count_nnz_rows_csr(csr_filepath):
    """
    Count the number of rows containing non-zero elements from a CSR file.
    
    Args:
        csr_filepath: Path to the CSR file
        
    Returns:
        int: Number of rows with at least one non-zero element
    """
    with open(csr_filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the indptr line
    indptr_line = None
    for line in lines:
        if line.startswith('indptr'):
            indptr_line = line.strip()
            break
    
    if indptr_line is None:
        return 0
        
    # Parse indptr array using parse_array from create_csr
    indptr = parse_array(indptr_line, "indptr", int)
    
        
    # Count rows with non-zero elements
    nnz_rows = 0
    for i in range(len(indptr) - 1):
        if indptr[i+1] - indptr[i] > 0:  # This row has at least one non-zero
            nnz_rows += 1
            
    return nnz_rows

def calculate_avg_nnz_per_row_csr(csr_filepath):
    """
    Calculate the average number of non-zero elements per non-zero row from a CSR file.
    
    Args:
        csr_filepath: Path to the CSR file
        
    Returns:
        float: Average number of non-zero elements per non-zero row
    """
    with open(csr_filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the indptr line
    indptr_line = None
    for line in lines:
        if line.startswith('indptr'):
            indptr_line = line.strip()
            break
    
    if indptr_line is None:
        return 0.0
        
    # Parse indptr array using parse_array from create_csr
    indptr = parse_array(indptr_line, "indptr", int)
    
    if len(indptr) < 2:
        return 0.0
        
    # Calculate total nnz and number of nnz rows
    total_nnz = indptr[-1]  # Last element of indptr is total nnz
    nnz_rows = 0
    for i in range(len(indptr) - 1):
        if indptr[i+1] - indptr[i] > 0:  # This row has at least one non-zero
            nnz_rows += 1
    
    if nnz_rows == 0:
        return 0.0
        
    return total_nnz / nnz_rows

def calculate_nnz_rows_slope(matrix_name, reduction_type="random"):
    """
    Calculate the slope of number of nnz rows vs density for a matrix.
    
    Args:
        matrix_name: Name of the matrix
        reduction_type: Type of reduction ("random", "truncated", "consec")
        
    Returns:
        float: Slope of the relationship (negative means fewer nnz rows as density decreases)
    """
    percentages = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    nnz_rows_counts = []
    
    # Count nnz rows for each density level
    for pct in percentages:
        if pct == 100:
            csr_file = f"csr_files/{matrix_name}.csr"
        else:
            csr_file = f"csr_files/{matrix_name}_{reduction_type}_{100-pct}pct.csr"
        
        nnz_rows = count_nnz_rows_csr(csr_file)
        nnz_rows_counts.append(nnz_rows)
    
    x = np.array(percentages)
    y = np.array(nnz_rows_counts)

    
    # Calculate slope using least squares
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    
    return slope

def calculate_avg_nnz_per_row_slope(matrix_name, reduction_type="random"):
    """
    Calculate the slope of average nnz per nnz row vs density for a matrix.
    
    Args:
        matrix_name: Name of the matrix
        reduction_type: Type of reduction ("random", "truncated", "consec")
        
    Returns:
        float: Slope of the relationship (negative means lower avg nnz per row as density decreases)
    """
    percentages = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    avg_nnz_per_row_values = []
    
    # Calculate avg nnz per row for each density level
    for pct in percentages:
        if pct == 100:
            csr_file = f"csr_files/{matrix_name}.csr"
        else:
            csr_file = f"csr_files/{matrix_name}_{reduction_type}_{100-pct}pct.csr"
        
        avg_nnz_per_row = calculate_avg_nnz_per_row_csr(csr_file)
        avg_nnz_per_row_values.append(avg_nnz_per_row)
    
    x = np.array(percentages)
    y = np.array(avg_nnz_per_row_values)

    
    # Calculate slope using least squares
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    
    return slope

if __name__ == "__main__":
    matrices_dir = 'matrices'
    results_dir = 'results'
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    dtypes = ['real', 'binary']
    
    with open("matrices.csv", "w") as fmatrix:
        fmatrix.write("matrix,high_br_mispreds,nnz_speedup,num_nnz_rows,is_scale_free,nnz_rows_slope,avg_nnz_per_row_slope\n")

        for dtype in dtypes:
            # Search for real and binary matrices with nnz between 20,000 and 20,000,000
            matrices = search(nzbounds=(2000000, 20000000), dtype=dtype, limit=10000)
            print(f"Found {len(matrices)} matrices with nnz between 20,000 and 20,000,000")

            for mat in matrices:
                print(f"\n{'='*60}")
                print(f"Processing Matrix: {mat.name} (ID: {mat.id}, NNZ: {mat.nnz})")
                print(f"{'='*60}")
                mtx_path = os.path.join(matrices_dir, f"{mat.name}.mtx")
                tar_path = os.path.join(matrices_dir, f"{mat.name}.tar.gz")
                mat.download(destpath=matrices_dir)
                print(f"Downloaded {mat.name}")

                with tarfile.open(tar_path, 'r:gz') as tar:
                    # Extract only the .mtx file
                    mtx_filename = f"{mat.name}.mtx"
                    try:
                        for member in tar.getmembers():
                            if member.name == f"{mat.name}/{mtx_filename}":
                                tar.extract(member, path=matrices_dir)
                                extracted_path = os.path.join(matrices_dir, member.name)
                                os.rename(extracted_path, mtx_path)
                                print(f"Extracted {member.name} as {mtx_filename}")
                                break
                    except EOFError:
                        print(f"Error extracting {mat.name}.tar.gz")
                        os.remove(tar_path)
                        continue

                # Delete the tar file after extraction
                os.remove(tar_path)
                os.rmdir(os.path.join(matrices_dir, mat.name))
                print(f"Deleted {mat.name}.tar.gz")
                print(f"Matrix file ready at {mtx_path}")

                num_nnz_rows = count_nnz_rows(mtx_path)
                scale_free = is_scale_free(mtx_path)

                create_csr_variants(mat.name)

                run_sparse_operation(mat.name, "SpMV", "random", 100, "br_mispreds")
                run_sparse_operation(mat.name, "SpMV", "random", 100, "timing")
                
                # Calculate slope of nnz rows vs density
                nnz_rows_slope = calculate_nnz_rows_slope(mat.name, "random")
                if nnz_rows_slope is not None:
                    print(f"NNZ rows slope: {nnz_rows_slope:.4f} (negative = fewer nnz rows as density decreases)")
                else:
                    print("Could not calculate NNZ rows slope")
                    nnz_rows_slope = "N/A"

                # Calculate slope of avg nnz per row vs density
                avg_nnz_per_row_slope = calculate_avg_nnz_per_row_slope(mat.name, "random")
                if avg_nnz_per_row_slope is not None:
                    print(f"Avg NNZ per row slope: {avg_nnz_per_row_slope:.4f} (negative = lower avg nnz per row as density decreases)")
                else:
                    print("Could not calculate avg NNZ per row slope")
                    avg_nnz_per_row_slope = "N/A"

                br_mispreds_result_file = os.path.join(results_dir, f"papi_{mat.name}_SpMV_random.csv")
                timing_result_file = os.path.join(results_dir, f"timing_{mat.name}_SpMV_random.csv")

                percentages = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
                mispreds = []
                timings = []
                with open(br_mispreds_result_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mispreds.append(float(row['branch_mispreds']))

                with open(timing_result_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        timings.append(float(row['time']))

                high_br_mispreds = any(mispred > 1.0 for mispred in mispreds)
                nnz_speedup = all(timings[i] > timings[i+1] for i in range(len(timings)-1))

                fmatrix.write(f"{mat.name},{high_br_mispreds},{nnz_speedup},{num_nnz_rows},{scale_free},{nnz_rows_slope},{avg_nnz_per_row_slope}\n")
                fmatrix.flush()

                # Delete the variants created in the csr_files directory
                for file in os.listdir(f"csr_files/"):
                    if file.endswith(".csr"):
                        os.remove(os.path.join(f"csr_files", file))

                # Also delete contents of Generated_dense_tensors directory
                for file in os.listdir(f"Generated_dense_tensors/"):
                    if file.endswith(".matrix") or file.endswith(".vector"):
                        os.remove(os.path.join(f"Generated_dense_tensors", file))
                        print(f"Deleted {file}")

                # Also delete the matrix file
                os.remove(mtx_path)
import os
import subprocess
import tarfile
from ssgetpy import search
from create_csr import create_csr_variants

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
EXEC_DIR = os.path.join(BASE_PATH, "spv8-public", "data")
MATRIX_DIR = os.path.join(BASE_PATH, "matrices")
CSR_FILES_DIR = os.path.join(BASE_PATH, "csr_files")

def extract_timing(output_text):
    """Extract timing information from the program output."""
    time_val = None
    
    for line in output_text.split('\n'):
        if "Time:" in line:
            # Extract the time value
            time_str = line.split(":")[-1].strip().split()[0]
            time_val = float(time_str)
        elif "Correct:" in line:
            # Extract correctness
            correct_str = line.split(":")[-1].strip()
            assert(correct_str.lower() == "true")
    
    return time_val

def write_per_matrix_csvs():
    results_file_path = os.path.join(BASE_PATH, "spv8_eval_results.txt")
    output_dir = os.path.join(BASE_PATH, "results")

    # Read and collect timings grouped by base matrix name
    grouped = {}
    if not os.path.exists(results_file_path):
        print(f"Results file not found: {results_file_path}. Skipping CSV creation.")
        return

    with open(results_file_path, "r") as fr:
        for line in fr:
            line = line.strip()
            if not line or line.startswith("Matrix"):
                continue
            # Expect lines like: name,time
            parts = line.split(",", 1)
            if len(parts) != 2:
                raise Exception(f"Malformed line in results file: {line}")
            name, time_str = parts[0].strip(), parts[1].strip()
            time_val = float(time_str)

            # Interpret names like `{matrix}_random_{num}pct` or bare `{matrix}`
            # Note: pct in filename means % removed, so actual % kept = 100 - pct
            if "_random_" in name:
                base, tail = name.split("_random_", 1)
                # tail expected like '90pct'
                if tail.endswith("pct"):
                    pct_str = tail[:-3]
                    try:
                        pct_removed = int(pct_str)
                        pct = 100 - pct_removed  # Convert to % kept
                    except ValueError:
                        # fallback: mark as unknown (skip)
                        continue
                else:
                    # no pct suffix: skip
                    continue
            else:
                base = name
                pct = 100

            grouped.setdefault(base, []).append((pct, time_val))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write per-matrix CSV files
    for base, entries in grouped.items():
        # sort by percentage descending so 100 comes first, then 90, etc.
        entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)
        # Sanitize base name to remove null bytes and other problematic characters
        sanitized_base = base.replace('\x00', '').replace('/', '_').replace('\\', '_')
        if not sanitized_base:
            print(f"Skipping invalid matrix name: {repr(base)}")
            continue
        out_name = f"timing_{sanitized_base}_SpMV_random.csv"
        out_path = os.path.join(output_dir, out_name)
        try:
            with open(out_path, "w") as fo:
                fo.write("Percentage,time\n")
                for pct, t in entries_sorted:
                    fo.write(f"{pct},{t:.6f}\n")
            print(f"Wrote {out_path} ({len(entries_sorted)} rows)")
        except (OSError, ValueError) as e:
            print(f"Failed to write {out_path}: {e}")

def run_generate_mtx():
    """Run the generate_mtx.py script to prepare data for spv8."""
    generate_mtx_script = os.path.join(BASE_PATH, "spv8-public", "contrib", "generate_mtx.py")
    result = subprocess.run(["python3", generate_mtx_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running generate_mtx.py: {result.stderr}")
        raise RuntimeError("generate_mtx.py failed")
    print("Successfully generated spv8 data files")

def cleanup_matrix_files(matrix_name):
    """Clean up downloaded matrix and generated files for a specific matrix."""
    # Remove from matrices directory
    mtx_path = os.path.join(MATRIX_DIR, f"{matrix_name}.mtx")
    if os.path.exists(mtx_path):
        os.remove(mtx_path)
        print(f"Deleted {mtx_path}")
    
    # Remove CSR variants
    for file in os.listdir(CSR_FILES_DIR):
        if file.startswith(matrix_name) and file.endswith(".mtx"):
            file_path = os.path.join(CSR_FILES_DIR, file)
            os.remove(file_path)
            print(f"Deleted {file_path}")
    
    # Remove generated dense tensors
    dense_tensor_dir = os.path.join(BASE_PATH, "Generated_dense_tensors")
    if os.path.exists(dense_tensor_dir):
        for file in os.listdir(dense_tensor_dir):
            if file.endswith(".matrix") or file.endswith(".vector"):
                file_path = os.path.join(dense_tensor_dir, file)
                os.remove(file_path)
                print(f"Deleted {file_path}")
    
    # Remove spv8 data directories for this matrix and its variants
    import shutil
    for dir_name in os.listdir(EXEC_DIR):
        if dir_name.startswith(matrix_name):
            data_dir = os.path.join(EXEC_DIR, dir_name)
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
                print(f"Deleted {data_dir}")


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(CSR_FILES_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "results"), exist_ok=True)

    dtypes = ['real', 'binary']
    
    with open("spv8_eval_results.txt", "w") as fresults:
        fresults.write("Matrix,Time\n")
        
        for dtype in dtypes:
            # Search for real and binary matrices with nnz between 20,000 and 20,000,000
            matrices = search(nzbounds=(20000, 20000000), dtype=dtype, limit=10000)
            print(f"Found {len(matrices)} {dtype} matrices with nnz between 20,000 and 20,000,000")

            for mat in matrices:
                matrix_name = mat.name
                print(f"\n{'='*60}")
                print(f"Processing Matrix: {matrix_name} (ID: {mat.id}, NNZ: {mat.nnz})")
                print(f"{'='*60}")
                
                # Download the matrix
                mtx_path = os.path.join(MATRIX_DIR, f"{matrix_name}.mtx")
                tar_path = os.path.join(MATRIX_DIR, f"{matrix_name}.tar.gz")
                
                try:
                    mat.download(destpath=MATRIX_DIR)
                    print(f"Downloaded {matrix_name}")

                    # Extract the .mtx file from the tar.gz
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        mtx_filename = f"{matrix_name}.mtx"
                        try:
                            for member in tar.getmembers():
                                if member.name == f"{matrix_name}/{mtx_filename}":
                                    tar.extract(member, path=MATRIX_DIR)
                                    extracted_path = os.path.join(MATRIX_DIR, member.name)
                                    os.rename(extracted_path, mtx_path)
                                    print(f"Extracted {member.name} as {mtx_filename}")
                                    break
                        except EOFError:
                            print(f"Error extracting {matrix_name}.tar.gz")
                            os.remove(tar_path)
                            continue

                    # Delete the tar file and extracted directory after extraction
                    os.remove(tar_path)
                    matrix_dir = os.path.join(MATRIX_DIR, matrix_name)
                    if os.path.exists(matrix_dir):
                        os.rmdir(matrix_dir)
                    print(f"Deleted {matrix_name}.tar.gz and extracted directory")

                except Exception as e:
                    print(f"Error downloading {matrix_name}: {e}")
                    continue

                # Create CSR variants
                try:
                    print(f"Creating CSR variants for {matrix_name}...")
                    create_csr_variants(matrix_name)
                    print(f"Successfully created CSR variants for {matrix_name}")
                except Exception as e:
                    print(f"Error creating CSR variants for {matrix_name}: {e}")
                    cleanup_matrix_files(matrix_name)
                    continue

                # Run generate_mtx.py to prepare spv8 data
                try:
                    print(f"Generating spv8 data for {matrix_name}...")
                    run_generate_mtx()
                except Exception as e:
                    print(f"Error generating spv8 data for {matrix_name}: {e}")
                    cleanup_matrix_files(matrix_name)
                    continue

                # Run spv8 evaluation for each variant
                for dir_name in os.listdir(EXEC_DIR):
                    # Check if it's a directory and matches the current matrix
                    dir_path = os.path.join(EXEC_DIR, dir_name)
                    if not os.path.isdir(dir_path):
                        continue
                    if not dir_name.startswith(matrix_name):
                        continue
                    
                    try:
                        cmd = ["../../bin/spmv_spv8", "100", "0", "1"]
                        result = subprocess.run(cmd, cwd=dir_path, capture_output=True, text=True, check=True)
                        
                        timing_info = extract_timing(result.stdout)
                        print("\n" + "=" * 60)
                        print("Matrix:", dir_name)
                        print("=" * 60)
                        print(f"{timing_info:.6f}")
                        print("=" * 60)
                        fresults.write(f"{dir_name},{timing_info:.6f}\n")
                        fresults.flush()
                    except Exception as e:
                        print(f"Error running spv8 for {dir_name}: {e}")
                        continue

                # Clean up after processing this matrix
                cleanup_matrix_files(matrix_name)
                
                # Write per-matrix CSVs after each matrix is processed
                write_per_matrix_csvs()
                
                # Clean up the results file to avoid processing the same matrix repeatedly
                results_file_path = os.path.join(BASE_PATH, "spv8_eval_results.txt")
                with open(results_file_path, "w") as f:
                    f.write("Matrix,Time\n")




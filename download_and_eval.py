import csv
import os
import tarfile

from ssgetpy import search

from create_csr import create_csr_variants
from eval import run_sparse_operation

if __name__ == "__main__":
    matrices_dir = 'matrices'
    results_dir = 'results'
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    dtypes = ['real', 'binary']
    
    with open("matrices.csv", "w") as fmatrix:
        fmatrix.write("matrix,high_br_mispreds,nnz_speedup,num_nnz_rows\n")

        for dtype in dtypes:
            # Search for real and binary matrices with nnz between 20,000 and 20,000,000
            matrices = search(nzbounds=(20000, 20000000), dtype=dtype, limit=10000)
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

                create_csr_variants(mat.name)

                run_sparse_operation(mat.name, "SpMV", "random", 100, "br_mispreds")
                run_sparse_operation(mat.name, "SpMV", "random", 100, "timing")

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

                fmatrix.write(f"{mat.name},{high_br_mispreds},{nnz_speedup}\n")
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
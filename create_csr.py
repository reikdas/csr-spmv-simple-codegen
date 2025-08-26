import scipy

def save_csr_to_file(csr_matrix, filename="foo.csr"):
    """Save CSR matrix data to a file."""
    csr_matrix = scipy.io.mmread("brainpc2.mtx")
    csr_matrix = csr_matrix.tocsr()
    try:
        with open(filename, 'w') as f:
            # Write indptr (row pointers)
            f.write("indptr=[")
            f.write(",".join(map(str, csr_matrix.indptr)))
            f.write("]\n")
            
            # Write indices (column indices)
            f.write("indices=[")
            f.write(",".join(map(str, csr_matrix.indices)))
            f.write("]\n")
            
            # Write data (matrix values)
            f.write("data=[")
            f.write(",".join(map(str, csr_matrix.data)))
            f.write("]\n")
            
        print(f"CSR matrix saved to {filename}")
        print(f"Matrix shape: {csr_matrix.shape}")
        print(f"Non-zeros: {csr_matrix.nnz}")
        
    except Exception as e:
        print(f"Error saving CSR matrix: {e}")

if __name__ == "__main__":
    save_csr_to_file("brainpc2.mtx")
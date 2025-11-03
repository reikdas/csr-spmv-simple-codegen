import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

# Matrix dimensions
rows = 100_000
cols = 1_000_000
nnz = 100_000

# Output directory
out_dir = "matrices"
os.makedirs(out_dir, exist_ok=True)

# --- 1. Single Row Matrix ---
row_index = 0
col_indices = np.random.choice(cols, size=nnz, replace=False)
vals_row = np.random.rand(nnz)
row_indices = np.full(nnz, row_index)

A_row = csr_matrix((vals_row, (row_indices, col_indices)), shape=(rows, cols))
row_path = os.path.join(out_dir, "singlerow.mtx")
mmwrite(row_path, A_row)

print(f"Row matrix saved as '{row_path}'")
print(f"  Shape: {A_row.shape}, nnz: {A_row.nnz}, density: {A_row.nnz / (A_row.shape[0] * A_row.shape[1]):.6f}")

# --- 2. Single Column Matrix ---
col_index = 0
row_indices = np.random.choice(rows, size=nnz, replace=False)
vals_col = np.random.rand(nnz)
col_indices = np.full(nnz, col_index)

A_col = csr_matrix((vals_col, (row_indices, col_indices)), shape=(rows, cols))
col_path = os.path.join(out_dir, "singlecol.mtx")
mmwrite(col_path, A_col)

print(f"Column matrix saved as '{col_path}'")
print(f"  Shape: {A_col.shape}, nnz: {A_col.nnz}, density: {A_col.nnz / (A_col.shape[0] * A_col.shape[1]):.6f}")

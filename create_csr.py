import os
import random
from pathlib import Path
from typing import Callable, List, Tuple, TypeVar

import scipy

T = TypeVar("T", int, float)

def parse_array(line: str, key: str, caster: Callable[[str], T]) -> List[T]:
    """Parse a line like 'key=[a,b,c,...]' and return typed list."""
    assert line.startswith(key + "=") or line.startswith(key + "=["), f"Line does not start with {key}="
    prefix, arr_str = line.split("=", 1)
    assert prefix.strip() == key, f"Unexpected key in line: {prefix}"
    assert arr_str.strip().startswith("["), "Array must start with '['"
    assert arr_str.strip().endswith("]"), "Array must end with ']'"
    body = arr_str.strip()[1:-1]
    # Safe parse by splitting on commas and converting
    parts = [] if body.strip() == "" else [p.strip() for p in body.split(",")]
    values: List[T] = []
    for p in parts:
        if p == "":
            continue
        values.append(caster(p))
    return values


def format_array(prefix: str, values: List[int] | List[float]) -> str:
    if all(isinstance(v, int) for v in values):
        content = ",".join(str(int(v)) for v in values)
    else:
        content = ",".join(repr(float(v)) for v in values)
    return f"{prefix}=[{content}]"


def load_csr_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_csr_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def find_line_index(lines: List[str], key: str) -> int:
    for i, line in enumerate(lines):
        if line.startswith(key + "=") or line.startswith(key + "=["):
            return i
    raise ValueError(f"Key {key} not found in file")


def sampled_removal_mask(indptr: List[int], drop_fraction: float, seed: int | None) -> List[bool]:
    """Return a mask for entries in the CSR data (length == indptr[-1]) marking which
    entries to keep (True) or drop (False). We drop approximately drop_fraction of entries.

    We sample uniformly at random over all nonzero positions, but maintain row structure:
    each row keeps a subset of its entries, and rows are allowed to become empty.
    """
    if not indptr:
        return []
    nnz = indptr[-1]
    num_drop = int(nnz * drop_fraction)
    num_drop = max(0, min(num_drop, nnz))
    rng = random.Random(seed)
    to_drop = set(rng.sample(range(nnz), num_drop)) if num_drop > 0 else set()
    mask = [i not in to_drop for i in range(nnz)]
    return mask


def apply_mask_to_csr(csr_val: List[float], indices: List[int], indptr: List[int], mask: List[bool]) -> Tuple[List[float], List[int], List[int]]:
    assert len(csr_val) == len(indices) == len(mask)
    n_rows = len(indptr) - 1
    new_val: List[float] = []
    new_indices: List[int] = []
    new_indptr: List[int] = [0]
    for r in range(n_rows):
        row_start = indptr[r]
        row_end = indptr[r + 1]
        kept_in_row = 0
        for i in range(row_start, row_end):
            if mask[i]:
                new_val.append(float(csr_val[i]))
                new_indices.append(int(indices[i]))
                kept_in_row += 1
        new_indptr.append(new_indptr[-1] + kept_in_row)
    assert new_indptr[-1] == len(new_val) == len(new_indices)
    return new_val, new_indices, new_indptr

def truncate_csr(csr_val: List[float], indices: List[int], indptr: List[int], keep_fraction: float) -> Tuple[List[float], List[int], List[int]]:
    """Truncate CSR data to keep only the first keep_fraction of elements."""
    nnz = len(csr_val)
    keep_count = int(nnz * keep_fraction)
    keep_count = max(0, min(keep_count, nnz))
    
    # Keep only the first keep_count elements
    new_val = csr_val[:keep_count]
    new_indices = indices[:keep_count]
    
    # Recalculate indptr based on the new data
    new_indptr = [0]
    current_pos = 0
    for r in range(len(indptr) - 1):
        row_start = indptr[r]
        row_end = indptr[r + 1]
        row_size = row_end - row_start
        
        # Count how many elements from this row we're keeping
        kept_in_row = min(row_size, max(0, keep_count - current_pos))
        new_indptr.append(new_indptr[-1] + kept_in_row)
        current_pos += kept_in_row
        
        if current_pos >= keep_count:
            break
    
    # Ensure we have the right number of rows
    while len(new_indptr) < len(indptr):
        new_indptr.append(new_indptr[-1])
    
    return new_val, new_indices, new_indptr

def truncate_consec_csr(csr_val: List[float], indices: List[int], indptr: List[int], keep_fraction: float) -> Tuple[List[float], List[int], List[int]]:
    """Truncate CSR data to keep only the first keep_fraction of elements."""
    nnz = len(csr_val)
    remove_count = int(nnz * (1 - keep_fraction))
    if remove_count == 0:
        return csr_val, indices, indptr
    # Choose a random consecutive block from the middle
    max_start = nnz - remove_count
    if max_start <= 0:
        start = 0
    else:
        # Avoid removing from the very start or end
        min_start = max(1, nnz // 4)
        max_start = min(max_start, nnz - remove_count - 1)
        if max_start <= min_start:
            start = min_start
        else:
            start = random.randint(min_start, max_start)
    end = start + remove_count
    # Remove the block
    new_val = csr_val[:start] + csr_val[end:]
    new_indices = indices[:start] + indices[end:]
    # Recalculate indptr
    new_indptr = [0]
    curr = 0
    for r in range(len(indptr) - 1):
        row_start = indptr[r]
        row_end = indptr[r + 1]
        row_nnz = row_end - row_start
        # Count how many nnz in this row are kept
        kept = 0
        for i in range(row_start, row_end):
            if i < start or i >= end:
                kept += 1
        new_indptr.append(new_indptr[-1] + kept)
    return new_val, new_indices, new_indptr


def push_zero_rows_to_end(A):
    import numpy as _np
    A = A.tocsr().sorted_indices()
    row_nnz = _np.diff(A.indptr)

    nonzero_rows = _np.flatnonzero(row_nnz > 0)
    zero_rows = _np.flatnonzero(row_nnz == 0)
    row_perm = _np.concatenate([nonzero_rows, zero_rows]).astype(int)

    A_re = A[row_perm, :]
    return A_re, row_perm

## Gray reordering 

def _gray_sequence(_nbits: int):
    for i in range(1 << _nbits):
        yield i ^ (i >> 1)

def _row_bitmap(_row_cols, _n_cols: int, _nbits: int) -> int:
    import numpy as _np
    if _row_cols.size == 0 or _n_cols == 0:
        return 0
    bins = (_row_cols.astype(_np.int64) * _nbits // max(1, _n_cols)).astype(_np.int64)
    bm = 0
    # unique to reduce ops
    for b in _np.unique(bins):
        b = int(max(0, min(int(b), _nbits - 1)))
        bm |= (1 << b)
    return bm

def gray_reorder_csr_rows_only(A, *, nbits: int = 16, dense_threshold: int = 20):
    """
    Bitmap-based Gray reordering (rows only).
      1) Build nbits-length bitmap per row over coarse column buckets.
      2) Group rows by identical bitmap.
      3) Emit groups in Gray-code order (successive bucket ids differ by 1 bit).
      4) Append 'dense' rows (nnz > dense_threshold) at the end (degree-desc stable).
    Returns (A_reordered, row_perm).
    """
    import numpy as _np
    from collections import defaultdict as _defaultdict

    if nbits <= 0:
        raise ValueError("nbits must be positive")
    A = A.tocsr().sorted_indices()
    m, n = A.shape
    ip, idx = A.indptr, A.indices
    row_nnz = _np.diff(ip)
    dense_mask = row_nnz > dense_threshold

    groups = _defaultdict(list)
    for i in range(m):
        if dense_mask[i]:
            continue
        s, e = ip[i], ip[i + 1]
        bm = _row_bitmap(idx[s:e], n, nbits)
        groups[bm].append(i)

    sparse_row_order = []
    for g in _gray_sequence(nbits):
        if g in groups:
            sparse_row_order.extend(groups[g])

    dense_rows = _np.flatnonzero(dense_mask)
    if dense_rows.size:
        dense_rows = dense_rows[_np.argsort(-row_nnz[dense_rows], kind="mergesort")]
        row_perm = _np.concatenate([_np.asarray(sparse_row_order, dtype=int), dense_rows.astype(int)])
    else:
        row_perm = _np.asarray(sparse_row_order, dtype=int)

    A_re = A[row_perm, :]
    return A_re, row_perm


## CSR writer functions
def save_csr_data(matrix_name: str, new_val: List[float], new_indices: List[int], new_indptr: List[int], csr_dir: str = "csr_files") -> None:
    """
    Write CSR data to file, optionally with zero rows moved to the end.
    
    Args:
        matrix_name: Name of the matrix (used for filename)
        new_val: CSR data values
        new_indices: CSR column indices  
        new_indptr: CSR row pointers
        csr_dir: Directory to save the file
        apply_zero_reorder: If True, apply zero-rows-to-end transformation
    """
    import os as _os
    import numpy as _np
    from scipy.sparse import csr_matrix as _csr
    
    if not _os.path.isdir(csr_dir):
        _os.makedirs(csr_dir, exist_ok=True)
    
    # Create scipy CSR matrix from the provided data
    m = len(new_indptr) - 1
    n = (max(new_indices) + 1) if len(new_indices) > 0 else 0
    A = _csr((_np.asarray(new_val, dtype=float),
                _np.asarray(new_indices, dtype=int),
                _np.asarray(new_indptr, dtype=int)),
               shape=(m, n))
    
    # Apply zero-rows-to-end transformation if requested
    A, _rowp0 = push_zero_rows_to_end(A)
    
    # Write to file
    out_path = _os.path.join(csr_dir, f"{matrix_name}.csr")
    with open(out_path, 'w') as f:
        # Write indptr (row pointers)
        f.write("indptr=[")
        f.write(",".join(map(str, A.indptr)))
        f.write("]\n")
        
        # Write indices (column indices)
        f.write("indices=[")
        f.write(",".join(map(str, A.indices)))
        f.write("]\n")
        
        # Write data (matrix values)
        f.write("data=[")
        f.write(",".join(map(str, A.data)))
        f.write("]\n")


def create_csr_variants(matrix):
    # Load the matrix and get CSR data
    csr_matrix = scipy.io.mmread(f"matrices/{matrix}.mtx")
    csr_matrix = csr_matrix.tocsr()
    
    # Extract CSR components
    csr_val = csr_matrix.data.tolist()
    indices = csr_matrix.indices.tolist()
    indptr = csr_matrix.indptr.tolist()
    
    # Save original CSR and reordered version
    # save_csr_to_file(matrix)
    save_csr_data(matrix, csr_val, indices, indptr, "csr_files")
    
    # Load the reordered version for further processing
    lines = load_csr_lines(f"csr_files/{matrix}.csr")
    idx_indptr = find_line_index(lines, "indptr")
    idx_indices = find_line_index(lines, "indices")
    idx_val = find_line_index(lines, "data")
    indptr = parse_array(lines[idx_indptr], "indptr", int)
    indices = parse_array(lines[idx_indices], "indices", int)
    csr_val = parse_array(lines[idx_val], "data", float)
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        fraction = pct / 100.0
        
        # Build mask and apply
        mask = sampled_removal_mask(indptr, fraction, 42) # Hardcoded seed for reproducibility
        new_val, new_indices, new_indptr = apply_mask_to_csr(csr_val, indices, indptr, mask)

        # Create output filename
        output_filename = f"csr_files/{matrix}_random_{pct}pct.csr"

        # Save the variant using save_csr_data
        save_csr_data(f"{matrix}_random_{pct}pct", new_val, new_indices, new_indptr, "csr_files")

        keep_fraction = 1.0 - fraction

        new_val, new_indices, new_indptr = truncate_csr(csr_val, indices, indptr, keep_fraction)
        output_filename = f"csr_files/{matrix}_truncated_{pct}pct.csr"
        
        # Save the variant using save_csr_data
        save_csr_data(f"{matrix}_truncated_{pct}pct", new_val, new_indices, new_indptr, "csr_files")

        new_val, new_indices, new_indptr = truncate_consec_csr(csr_val, indices, indptr, keep_fraction)
        output_filename = f"csr_files/{matrix}_consec_{pct}pct.csr"
        
        # Save the variant using save_csr_data
        save_csr_data(f"{matrix}_consec_{pct}pct", new_val, new_indices, new_indptr, "csr_files")


if __name__ == "__main__":
    matrices = [p.stem for p in Path("matrices").glob("*.mtx")]
    for matrix in matrices:
        create_csr_variants(matrix)

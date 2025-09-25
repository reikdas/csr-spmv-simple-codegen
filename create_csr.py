import scipy
import random
from typing import List, Tuple, TypeVar, Callable
import os
from pathlib import Path


T = TypeVar("T", int, float)

def parse_array(line: str, key: str, caster: Callable[[str], T]) -> Tuple[str, List[T]]:
    """Parse a line like 'key=[a,b,c,...]' and return (prefix, typed list)."""
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

def save_csr_to_file(matrix_name):
    if not os.path.exists("csr_files"):
        os.mkdir("csr_files")
    csr_matrix = scipy.io.mmread(f"matrices/{matrix_name}.mtx")
    csr_matrix = csr_matrix.tocsr()
    try:
        with open(f"csr_files/{matrix_name}.csr", 'w') as f:
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
        
    except Exception as e:
        print(f"Error saving CSR matrix: {e}")


def _read_csr_text_to_scipy(path: str):
    import numpy as _np
    from scipy.sparse import csr_matrix as _csr

    lines = load_csr_lines(path)
    idx_indptr = find_line_index(lines, "indptr")
    idx_indices = find_line_index(lines, "indices")
    idx_val = find_line_index(lines, "data")

    indptr = parse_array(lines[idx_indptr], "indptr", int)
    indices = parse_array(lines[idx_indices], "indices", int)
    data = parse_array(lines[idx_val], "data", float)

    if len(indptr) == 0:
        return _csr((_np.array([], dtype=float), _np.array([], dtype=int), _np.array([0], dtype=int)), shape=(0, 0))

    m = len(indptr) - 1
    n = (max(indices) + 1) if len(indices) > 0 else 0

    A = _csr((_np.asarray(data, dtype=float),
              _np.asarray(indices, dtype=int),
              _np.asarray(indptr, dtype=int)),
             shape=(m, n))
    return A.tocsr().sorted_indices(), lines, (idx_indptr, idx_indices, idx_val)


def _write_scipy_to_csr_text(original_lines, idx_triplet, A, out_path: str) -> None:
    A = A.tocsr().sorted_indices()
    idx_indptr, idx_indices, idx_val = idx_triplet
    new_lines = original_lines.copy()
    new_lines[idx_indptr] = format_array("indptr", A.indptr.tolist())
    new_lines[idx_indices] = format_array("indices", A.indices.tolist())
    new_lines[idx_val] = format_array("data", A.data.astype(float).tolist())
    write_csr_lines(out_path, new_lines)


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


## gray and zeros to end csr writer.
def save_reordered_csrs(matrix_name: str, csr_dir: str = "csr_files", *, nbits: int = 16, dense_threshold: int = 20) -> None:
    """
    write two new files next to it:
      - *_gray.csr            (Gray row reordering)
      - *_zeros_to_end.csr    (zero rows moved to the end)
    """
    import os as _os
    if not _os.path.isdir(csr_dir):
        return

    for fname in _os.listdir(csr_dir):
        if not fname.startswith(matrix_name) or not fname.endswith(".csr"):
            continue
        if fname.endswith("_gray.csr") or fname.endswith("_zeros_to_end.csr"):
            # don't reprocess generated files
            continue

        src_path = _os.path.join(csr_dir, fname)
        A, lines, idx_triplet = _read_csr_text_to_scipy(src_path)

        # 1) Gray (rows only)
        A_gray, _rowp = gray_reorder_csr_rows_only(A, nbits=nbits, dense_threshold=dense_threshold)
        out_gray = _os.path.join(csr_dir, fname[:-4] + "_gray.csr")
        if not _os.path.exists(out_gray):
            _write_scipy_to_csr_text(lines, idx_triplet, A_gray, out_gray)

        # 2) Zero-rows-to-end
        A_zero, _rowp0 = push_zero_rows_to_end(A)
        out_zero = _os.path.join(csr_dir, fname[:-4] + "_zeros_to_end.csr")
        if not _os.path.exists(out_zero):
            _write_scipy_to_csr_text(lines, idx_triplet, A_zero, out_zero)



def test_reorderings_local(mtx_path: str, *, nbits: int = 16, dense_threshold: int = 20,
                           repeats: int = 20, trials: int = 5) -> None:
    """
    Load a MatrixMarket .mtx, run Gray & ZeroEnd reorderings,
    check algebra correctness with the correct permutation logic, and print robust micro-benchmarks.
    Speedup is defined as (orig_time / reordered_time). >1.00 means reordered is faster.
    """
    import time as _time
    import numpy as _np
    from statistics import median as _median
    from scipy.io import mmread as _mmread

    def _bench_pair(A, B, x, repeats: int, trials: int):
        """Paired benchmark: for each trial, time A and B back-to-back to reduce drift.
        Returns (median_time_A, median_time_B)."""
        # warmup both
        _ = A @ x; _ = B @ x
        times_A, times_B = [], []
        for _ in range(trials):
            # A
            tA = 0.0
            for __ in range(repeats):
                s = _time.perf_counter(); _ = A @ x; tA += _time.perf_counter() - s
            times_A.append(tA / repeats)
            # B
            tB = 0.0
            for __ in range(repeats):
                s = _time.perf_counter(); _ = B @ x; tB += _time.perf_counter() - s
            times_B.append(tB / repeats)
        return _median(times_A), _median(times_B)

    A = _mmread(mtx_path).tocsr().sorted_indices()
    m, n, nnz = A.shape[0], A.shape[1], A.nnz
    zero_rows = int(_np.sum(_np.diff(A.indptr) == 0))
    print(f"Loaded {mtx_path}: shape={m}x{n}, nnz={nnz}, zero_rows={zero_rows}")

    rng = _np.random.default_rng(0)
    x = rng.random(n)
    y0 = A @ x

    # ---- Gray (rows only) ----
    Ag, rowp = gray_reorder_csr_rows_only(A, nbits=nbits, dense_threshold=dense_threshold)
    yg = Ag @ x
    ok_g = _np.allclose(yg, y0[rowp])
    print(f"[Gray] SpMV correctness vs permuted reference: {ok_g}")

    tA_med, tG_med = _bench_pair(A, Ag, x, repeats=repeats, trials=trials)
    print(f"[Gray] median orig={tA_med:.6f}s, median gray={tG_med:.6f}s, speedup={tA_med/max(tG_med,1e-12):.2f}x")

    # ---- Zero rows to end ----
    Az, rowp0 = push_zero_rows_to_end(A)
    yz = Az @ x
    ok_z = _np.allclose(yz, y0[rowp0])
    print(f"[ZeroEnd] SpMV correctness vs permuted reference: {ok_z}")

    tA_med2, tZ_med = _bench_pair(A, Az, x, repeats=repeats, trials=trials)
    print(f"[ZeroEnd] median orig={tA_med2:.6f}s, median zeroend={tZ_med:.6f}s, speedup={tA_med2/max(tZ_med,1e-12):.2f}x")

    if zero_rows == 0:
        print("Note: matrix has no zero rows; ZeroEnd is (near) identity, so speedup should be ~1.00×.")



if __name__ == "__main__":
    matrices = [p.stem for p in Path("matrices").glob("*.mtx")]
    for matrix in matrices:
        save_reordered_csrs(matrix, csr_dir="csr_files", nbits=16, dense_threshold=20)
        save_csr_to_file(matrix)
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

            # Create a copy of lines for this variant
            variant_lines = lines.copy()
            
            # Replace lines; keep original formatting/order otherwise
            variant_lines[idx_val] = format_array("data", new_val)
            variant_lines[idx_indices] = format_array("indices", new_indices)
            variant_lines[idx_indptr] = format_array("indptr", new_indptr)
            write_csr_lines(output_filename, variant_lines)

            keep_fraction = 1.0 - fraction

            new_val, new_indices, new_indptr = truncate_csr(csr_val, indices, indptr, keep_fraction)
            output_filename = f"csr_files/{matrix}_truncated_{pct}pct.csr"
            variant_lines = lines.copy()
            variant_lines[idx_val] = format_array("data", new_val)
            variant_lines[idx_indices] = format_array("indices", new_indices)
            variant_lines[idx_indptr] = format_array("indptr", new_indptr)
            write_csr_lines(output_filename, variant_lines)

            new_val, new_indices, new_indptr = truncate_consec_csr(csr_val, indices, indptr, keep_fraction)
            output_filename = f"csr_files/{matrix}_consec_{pct}pct.csr"
            variant_lines = lines.copy()
            variant_lines[idx_val] = format_array("data", new_val)
            variant_lines[idx_indices] = format_array("indices", new_indices)
            variant_lines[idx_indptr] = format_array("indptr", new_indptr)
            write_csr_lines(output_filename, variant_lines)

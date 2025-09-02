import scipy
import random
from typing import List, Tuple, TypeVar, Callable
import os


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

if __name__ == "__main__":
    matrices = ["brainpc2", "heart1", "lowThrust_7"]
    for matrix in matrices:
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

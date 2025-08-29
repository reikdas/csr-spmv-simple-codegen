import os
from typing import List, Tuple
from create_csr import load_csr_lines, parse_array, find_line_index


def load_csr_file(filepath: str) -> Tuple[List[int], List[int], List[float]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSR file not found: {filepath}")
    
    lines = load_csr_lines(filepath)
    idx_indptr = find_line_index(lines, "indptr")
    idx_indices = find_line_index(lines, "indices")
    idx_val = find_line_index(lines, "data")
    
    indptr = parse_array(lines[idx_indptr], "indptr", int)
    indices = parse_array(lines[idx_indices], "indices", int)
    data = parse_array(lines[idx_val], "data", float)
    
    return indptr, indices, data


def count_nnz_elements(indptr: List[int]) -> int:
    return indptr[-1] - indptr[0]


def test_truncate_csr_reduction():
    original_file = "csr_files/brainpc2.csr"
    print(f"Loading original file: {original_file}")
    
    try:
        original_indptr, original_indices, original_data = load_csr_file(original_file)
        original_nnz = len(original_data)
        print(f"Original nnz: {original_nnz}")
        print(f"Original indptr length: {len(original_indptr)}")
        print(f"Original indices length: {len(original_indices)}")
        print(f"Original data length: {len(original_data)}")
        print()
    except Exception as e:
        print(f"Error loading original file: {e}")
        return False
    
    # Test percentages from 10% to 90% reduction
    test_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    all_tests_passed = True
    
    for pct in test_percentages:
        reduced_file = f"csr_files/brainpc2_reduced_{pct}pct.csr"
        expected_keep_fraction = (100 - pct) / 100.0
        expected_nnz = int(original_nnz * expected_keep_fraction)
        
        print(f"Testing {pct}% reduction (should keep {expected_keep_fraction:.1%} of elements)")
        print(f"Expected nnz: {expected_nnz}")
        
        try:
            # Load the reduced file
            reduced_indptr, reduced_indices, reduced_data = load_csr_file(reduced_file)
            actual_nnz = len(reduced_data)
            
            print(f"Actual nnz: {actual_nnz}")
            print(f"Reduced indptr length: {len(reduced_indptr)}")
            print(f"Reduced indices length: {len(reduced_indices)}")
            print(f"Reduced data length: {len(reduced_data)}")
            
            # Verify the reduction
            if actual_nnz == expected_nnz:
                print(f"✓ PASS: {pct}% reduction correct")
            else:
                print(f"✗ FAIL: {pct}% reduction incorrect")
                print(f"  Expected: {expected_nnz}, Got: {actual_nnz}")
                all_tests_passed = False
            
            # Additional checks
            if len(reduced_indices) != actual_nnz:
                print(f"✗ FAIL: Indices length mismatch. Expected: {actual_nnz}, Got: {len(reduced_indices)}")
                all_tests_passed = False
            
            if len(reduced_data) != actual_nnz:
                print(f"✗ FAIL: Data length mismatch. Expected: {actual_nnz}, Got: {len(reduced_data)}")
                all_tests_passed = False
            
            # Check that indptr is consistent
            if reduced_indptr[-1] != actual_nnz:
                print(f"✗ FAIL: Indptr inconsistency. Last value should be {actual_nnz}, got {reduced_indptr[-1]}")
                all_tests_passed = False
            
            print()
            
        except Exception as e:
            print(f"✗ FAIL: Error processing {reduced_file}: {e}")
            all_tests_passed = False
            print()
    
    # Summary
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! truncate_csr is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! There are issues with truncate_csr.")
    
    return all_tests_passed

if __name__ == "__main__":
    test_truncate_csr_reduction()
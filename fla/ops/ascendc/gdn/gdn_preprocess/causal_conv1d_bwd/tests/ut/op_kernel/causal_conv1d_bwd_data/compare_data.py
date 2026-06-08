#!/usr/bin/env python3
"""Compare kernel output against golden reference for causal_conv1d_bwd."""

import numpy as np
import sys


def compare_data(dtype="float32"):
    np_type = np.float32 if dtype == "float32" else np.float16
    
    # Read dimensions
    with open("dims.txt", "r") as f:
        parts = f.read().strip().split()
        B, T, D, W, activation = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    
    # Read output and golden
    dx_out = np.fromfile(f"{dtype}_dx_output.bin", dtype=np_type)
    dw_out = np.fromfile(f"{dtype}_dw_output.bin", dtype=np_type)
    db_out = np.fromfile(f"{dtype}_db_output.bin", dtype=np_type)
    
    dx_golden = np.fromfile(f"{dtype}_dx_golden.bin", dtype=np_type)
    dw_golden = np.fromfile(f"{dtype}_dw_golden.bin", dtype=np_type)
    db_golden = np.fromfile(f"{dtype}_db_golden.bin", dtype=np_type)
    
    dx_out = dx_out.reshape(B, T, D).astype(np.float32)
    dw_out = dw_out.reshape(W, D).astype(np.float32)
    db_out = db_out.reshape(D).astype(np.float32)
    
    dx_golden = dx_golden.reshape(B, T, D).astype(np.float32)
    dw_golden = dw_golden.reshape(W, D).astype(np.float32)
    db_golden = db_golden.reshape(D).astype(np.float32)
    
    rtol = 1e-3
    atol = 1e-5
    
    all_passed = True
    
    # Check dx
    if np.allclose(dx_out, dx_golden, rtol=rtol, atol=atol):
        print("dx: PASSED")
    else:
        diff = np.abs(dx_out - dx_golden)
        max_diff = np.max(diff)
        max_rel = np.max(diff / (np.abs(dx_golden) + 1e-8))
        mismatch = np.sum(~np.isclose(dx_out, dx_golden, rtol=rtol, atol=atol))
        print(f"dx: FAILED - max_abs_diff={max_diff:.6f}, max_rel_diff={max_rel:.6f}, mismatch_count={mismatch}/{dx_golden.size}")
        all_passed = False
    
    # Check dw
    if np.allclose(dw_out, dw_golden, rtol=rtol, atol=atol):
        print("dw: PASSED")
    else:
        diff = np.abs(dw_out - dw_golden)
        max_diff = np.max(diff)
        max_rel = np.max(diff / (np.abs(dw_golden) + 1e-8))
        mismatch = np.sum(~np.isclose(dw_out, dw_golden, rtol=rtol, atol=atol))
        print(f"dw: FAILED - max_abs_diff={max_diff:.6f}, max_rel_diff={max_rel:.6f}, mismatch_count={mismatch}/{dw_golden.size}")
        all_passed = False
    
    # Check db
    if np.allclose(db_out, db_golden, rtol=rtol, atol=atol):
        print("db: PASSED")
    else:
        diff = np.abs(db_out - db_golden)
        max_diff = np.max(diff)
        max_rel = np.max(diff / (np.abs(db_golden) + 1e-8))
        mismatch = np.sum(~np.isclose(db_out, db_golden, rtol=rtol, atol=atol))
        print(f"db: FAILED - max_abs_diff={max_diff:.6f}, max_rel_diff={max_rel:.6f}, mismatch_count={mismatch}/{db_golden.size}")
        all_passed = False
    
    if all_passed:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    dtype = sys.argv[1] if len(sys.argv) > 1 else "float32"
    sys.exit(compare_data(dtype))
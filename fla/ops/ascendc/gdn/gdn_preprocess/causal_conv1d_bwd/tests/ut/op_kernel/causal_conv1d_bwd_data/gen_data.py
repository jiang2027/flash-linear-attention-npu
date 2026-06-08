#!/usr/bin/env python3
"""Generate test input data and CPU golden reference for causal_conv1d_bwd kernel UT."""

import numpy as np
import sys
import os


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def causal_conv1d_bwd_cpu(x, weight, dy, y=None, activation=0):
    """CPU reference implementation of causal_conv1d_bwd.
    
    Args:
        x: [B, T, D] input
        weight: [W, D] weight
        dy: [B, T, D] output gradient
        y: [B, T, D] forward output (for silu activation)
        activation: 0=none, 1=silu, 2=swish
    Returns:
        dx: [B, T, D]
        dw: [W, D]
        db: [D]
    """
    B, T, D = x.shape
    W = weight.shape[0]
    
    dx = np.zeros((B, T, D), dtype=np.float32)
    dw = np.zeros((W, D), dtype=np.float32)
    db = np.zeros((D,), dtype=np.float32)
    
    for b in range(B):
        for t in range(T):
            for i_w in range(W):
                t_dy = t + i_w
                if t_dy < T:
                    dy_val = dy[b, t_dy, :]
                    if activation == 1 or activation == 2:
                        y_val = y[b, t_dy, :]
                        sig = sigmoid(y_val)
                        dy_val = dy_val * sig * (1.0 + y_val * (1.0 - sig))
                    
                    w_idx = W - 1 - i_w
                    dx[b, t, :] += dy_val * weight[w_idx, :]
                    dw[w_idx, :] += dy_val * x[b, t, :]
                    if i_w == 0:
                        db[:] += dy_val
    
    return dx, dw, db


def gen_data(shape_str, d_type="float32", activation=0):
    """Generate test data and golden reference.
    
    Args:
        shape_str: e.g. '(2, 128, 256, 4)' for (B, T, D, W)
        d_type: 'float32' or 'float16'
        activation: 0=none, 1=silu
    """
    shape = parse_shape(shape_str)
    B, T, D, W = shape
    np_type = np.float32 if d_type == "float32" else np.float16
    
    np.random.seed(42)
    x = np.random.uniform(-1.0, 1.0, (B, T, D)).astype(np_type)
    weight = np.random.uniform(-0.5, 0.5, (W, D)).astype(np_type)
    dy = np.random.uniform(-1.0, 1.0, (B, T, D)).astype(np_type)
    
    if activation == 1:
        y = np.random.uniform(-1.0, 1.0, (B, T, D)).astype(np_type)
        y.tofile(f"{d_type}_y.bin")
    else:
        y = None
    
    # CPU reference
    dx_golden, dw_golden, db_golden = causal_conv1d_bwd_cpu(
        x.astype(np.float32), weight.astype(np.float32),
        dy.astype(np.float32), y.astype(np.float32) if y is not None else None,
        activation)
    
    x.tofile(f"{d_type}_x.bin")
    weight.tofile(f"{d_type}_weight.bin")
    dy.tofile(f"{d_type}_dy.bin")
    dx_golden.astype(np_type).tofile(f"{d_type}_dx_golden.bin")
    dw_golden.astype(np_type).tofile(f"{d_type}_dw_golden.bin")
    db_golden.astype(np_type).tofile(f"{d_type}_db_golden.bin")
    
    # Also save dims in a text file
    with open("dims.txt", "w") as f:
        f.write(f"{B} {T} {D} {W} {activation}\n")
    
    print(f"Generated data: B={B} T={T} D={D} W={W} activation={activation}")
    return 0


def parse_shape(s):
    s = s.strip("()[]")
    parts = [int(x.strip()) for x in s.split(",")]
    return tuple(parts)


if __name__ == "__main__":
    shape_str = sys.argv[1] if len(sys.argv) > 1 else "(2, 128, 256, 4)"
    dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"
    activation = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sys.exit(gen_data(shape_str, dtype, activation))
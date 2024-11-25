import triton
import triton.language as tl
import torch

@triton.jit
def layernorm_forward(Y, Y_row_stride, X, X_row_stride, W, b, r, mu, n_cols,
    eps, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    mean_X = tl.sum(X_row, axis=0) / n_cols
    XX = X_row - mean_X
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = XX * inv_var * W_row + b_row
    tl.store(Y + col_offsets, output, mask=mask)

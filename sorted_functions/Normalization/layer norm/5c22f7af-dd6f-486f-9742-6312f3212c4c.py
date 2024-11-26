import triton
import triton.language as tl
import torch

@triton.jit
def layernorm_backward(dY, dY_row_stride, X, X_row_stride, W, b, r, mu,
    n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0)
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    inv_var = tl.load(r)
    mean = tl.load(mu)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = dY_W - tl.sum(dY_W, axis=0) / n_cols - normed * tl.sum(dY_W *
        normed, axis=0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dY + col_offsets, dX_row, mask=mask)

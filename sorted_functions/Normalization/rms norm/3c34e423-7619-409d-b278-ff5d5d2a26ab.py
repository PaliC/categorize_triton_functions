import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_forward(Y, Y_row_stride, X, X_row_stride, W, r, n_cols, eps,
    BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr = Y + row_idx * Y_row_stride
    X_ptr = X + row_idx * X_row_stride
    r_ptr = r + row_idx
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    X_squared = X_row * X_row
    mean_X_squared = tl.sum(X_squared, axis=0) / n_cols
    rms = tl.math.rsqrt(mean_X_squared + eps)
    tl.store(r_ptr, rms)
    output = X_row * rms * W_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)

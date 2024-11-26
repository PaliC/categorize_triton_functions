import triton
import triton.language as tl
import torch

@triton.jit
def _rms_layernorm_forward(Y, Y_row_stride, X, X_row_stride, W,
    W_row_stride, r, r_row_stride, n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)

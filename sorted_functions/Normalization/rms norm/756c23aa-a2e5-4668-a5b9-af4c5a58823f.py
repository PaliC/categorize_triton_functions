import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride,
    W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride, n_cols, eps, offset,
    casting_mode: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row
        X_row = X_row
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps
        offset = offset
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr, rstd)
    X_row = X_row * rstd
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row
    Y_row = X_row * (offset + W_row)
    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)

import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride,
    W_ptr, W_row_stride, B_ptr, B_row_stride, Mean_ptr, Mean_row_stride,
    RSTD_ptr, RSTD_row_stride, n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / n_cols
    rstd = rsqrt(var + eps)
    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)
    Y_row = (X_row - mean) * rstd * W_row + B_row
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)

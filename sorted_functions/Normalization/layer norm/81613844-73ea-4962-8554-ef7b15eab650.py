import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_fused_multi_pass(output_ptr, a_ptr, weight_ptr,
    bias_ptr, mean_ptr, rstd_ptr, output_row_stride, output_col_stride,
    a_row_stride, a_col_stride, N_SIZE, eps, IS_RMSNORM: 'tl.constexpr',
    HAS_BIAS: 'tl.constexpr', BLOCK_N_SIZE: 'tl.constexpr'):
    """
    Implementation from triton tutorial:
    https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py
    It requires multiple passes on the data to compute mean and variance, it is slower than the single pass version.
    -> only used in benchmarks
    """
    row_idx = tl.program_id(0)
    row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)
    mean_acc = tl.zeros((BLOCK_N_SIZE,), dtype=tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        cols_offs = block_n_start_idx + block_range_offs
        a = tl.load(a_ptr + row_off + cols_offs * a_col_stride, mask=
            cols_offs < N_SIZE, other=0.0, eviction_policy='evict_last')
        mean_acc += a
    mean = tl.sum(mean_acc, axis=0) / N_SIZE
    var_acc = tl.zeros((BLOCK_N_SIZE,), dtype=tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        cols_offs = block_n_start_idx + block_range_offs
        a = tl.load(a_ptr + row_off + cols_offs * a_col_stride, mask=
            cols_offs < N_SIZE, other=0.0, eviction_policy='evict_last')
        a = tl.where(cols_offs < N_SIZE, a - mean, 0.0)
        var_acc += a * a
    var = tl.sum(var_acc, axis=0) / N_SIZE
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        cols_offs = block_n_start_idx + tl.arange(0, BLOCK_N_SIZE)
        mask_ptr = cols_offs < N_SIZE
        weight = tl.load(weight_ptr + cols_offs, mask=mask_ptr)
        bias = tl.load(bias_ptr + cols_offs, mask=mask_ptr)
        a = tl.load(a_ptr + row_off + cols_offs * a_col_stride, mask=
            mask_ptr, other=0.0, eviction_policy='evict_first')
        a_hat = (a - mean) * rstd
        output = a_hat * weight + bias
        tl.store(output_ptr + row_idx * output_row_stride + cols_offs *
            output_col_stride, output, mask=mask_ptr)

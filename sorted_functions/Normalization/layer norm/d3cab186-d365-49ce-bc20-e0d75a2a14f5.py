import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_fused_single_pass(output_ptr, a_ptr, weight_ptr,
    bias_ptr, mean_ptr, rstd_ptr, output_row_stride, output_col_stride,
    a_row_stride, a_col_stride, N_SIZE, eps, HAS_BIAS: 'tl.constexpr',
    IS_RMSNORM: 'tl.constexpr', BLOCK_N_SIZE: 'tl.constexpr'):
    """
    Layernorm based on Welford's variance computation algorithm.
    https://changyaochen.github.io/welford/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    :param output_ptr: output tensor
    :param a_ptr: input tensor
    :param weight_ptr: weights applied to the normalized input
    :param bias_ptr: bias added to the normalized input
    :param mean_ptr: save mean tensor for backward
    :param rstd_ptr: save standard deviation tensor for backward
    :param a_row_stride: stride of the input tensor
    :param N_SIZE: number of elements per row in the input tensor
    :param eps: epsilon value to avoid division by zero
    :param HAS_BIAS: whether the bias is provided
    :param IS_RMSNORM: whether the normalization is rmsnorm (False == layernorm)
    :param BLOCK_N_SIZE: number of threads per block
    :return: None
    """
    row_idx = tl.program_id(0)
    a_row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)
    mean = 0.0
    var = 0.0
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        n_end_off = min(block_n_start_idx + BLOCK_N_SIZE, N_SIZE)
        block_cols_count = n_end_off - block_n_start_idx
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        a = tl.load(a_ptr + a_row_off + col_offs * a_col_stride, mask=
            a_ptr_mask, other=0.0, eviction_policy='evict_last')
        if IS_RMSNORM:
            var += tl.sum(a * a, axis=0)
        else:
            block_mean = tl.sum(a, axis=0) / block_cols_count
            delta_mean = block_mean - mean
            delta_mean_sqr = delta_mean * delta_mean
            block_delta = tl.sum((a - block_mean) * a, axis=0)
            mean += tl.sum((a - mean) * a_ptr_mask, axis=0) / n_end_off
            var += block_delta + delta_mean_sqr * (block_n_start_idx *
                block_cols_count) / n_end_off
    var /= N_SIZE
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        weight = tl.load(weight_ptr + col_offs, mask=a_ptr_mask)
        a = tl.load(a_ptr + a_row_off + col_offs * a_col_stride, mask=
            a_ptr_mask, other=0.0, eviction_policy='evict_first')
        a_hat = (a - mean) * rstd
        out = a_hat * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + col_offs, mask=a_ptr_mask)
            out = out + bias
        tl.store(output_ptr + row_idx * output_row_stride + col_offs *
            output_col_stride, out, mask=a_ptr_mask)

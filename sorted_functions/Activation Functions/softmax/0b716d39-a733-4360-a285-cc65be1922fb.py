import triton
import triton.language as tl
import torch

@triton.jit
def _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs,
    col_offsets, n_cols, use_mask: 'tl.constexpr', use_bias: 'tl.constexpr'):
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    if use_bias:
        bias = tl.load(bias_ptrs, mask=col_offsets < n_cols, other=float(
            '-inf'))
        row += bias
    if use_mask:
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float(
            '-inf'))
        row = tl.where(mask == 0, float('-1e20'), row)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

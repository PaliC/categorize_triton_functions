import triton
import triton.language as tl
import torch

@triton.jit
def softmax_mask_bias_kernel_two_rows(output_ptr, input_ptr, mask_ptr,
    bias_ptr, input_row_stride, output_row_stride, n_cols, n_heads,
    BLOCK_SIZE: 'tl.constexpr', use_mask: 'tl.constexpr', use_bias:
    'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_row_ptr = input_ptr + 2 * row_idx * input_row_stride
    output_row_ptr = output_ptr + 2 * row_idx * output_row_stride
    input_ptrs = input_row_ptr + col_offsets
    output_ptrs = output_row_ptr + col_offsets
    mask_ptrs = input_ptrs
    if use_mask:
        mask_row_ptr = mask_ptr + 2 * row_idx // (n_heads * n_cols) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets
    bias_ptrs = input_ptrs
    if use_bias:
        bias_row_ptr = bias_ptr + 2 * row_idx % (n_heads * n_cols) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets
    _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs,
        col_offsets, n_cols, use_mask, use_bias)
    mask_ptrs = input_ptrs
    if use_mask:
        mask_row_ptr = mask_ptr + (2 * row_idx + 1) // (n_heads * n_cols
            ) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets
    bias_ptrs = input_ptrs
    if use_bias:
        bias_row_ptr = bias_ptr + (2 * row_idx + 1) % (n_heads * n_cols
            ) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets
    _softmax_core(input_ptrs + n_cols, output_ptrs + n_cols, mask_ptrs,
        bias_ptrs, col_offsets, n_cols, use_mask, use_bias)

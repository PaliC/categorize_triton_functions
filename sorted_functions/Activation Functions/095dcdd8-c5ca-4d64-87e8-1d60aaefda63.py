import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel_forward(output_ptr, input_ptr, input_row_stride,
    output_row_stride, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    causal_mask = col_offsets > row_idx % n_cols
    row = row + tl.where(causal_mask, -float('inf'), 0.0)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

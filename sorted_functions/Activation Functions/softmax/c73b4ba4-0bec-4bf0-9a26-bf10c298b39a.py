import triton
import triton.language as tl
import torch

@triton.jit
def _softmax_kernel_fwd(input_ptr, stride_input_row, output_ptr,
    stride_output_row, num_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_id = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_id * stride_input_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_pointers = row_start_ptr + col_offsets
    row_data_mask = col_offsets < num_cols
    x = tl.load(input_pointers, mask=row_data_mask, other=0.0)
    safe_row = x - tl.max(x, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    output_row_ptr = output_ptr + row_id * stride_input_row
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, softmax_out, mask=row_data_mask)

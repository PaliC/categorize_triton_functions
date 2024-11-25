import triton
import triton.language as tl
import torch

@triton.jit
def softmax_grad_kernel_two_rows(d_output_ptr, output_ptr, d_input_ptr,
    d_output_row_stride, output_row_stride, d_input_row_stride, n_cols,
    BLOCK_SIZE: 'tl.constexpr', is_bf16: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_row_ptr = output_ptr + 2 * row_idx * output_row_stride
    d_output_row_ptr = d_output_ptr + 2 * row_idx * d_output_row_stride
    d_input_row_ptr = d_input_ptr + 2 * row_idx * d_input_row_stride
    output_ptrs = output_row_ptr + col_offsets
    d_output_ptrs = d_output_row_ptr + col_offsets
    d_input_ptrs = d_input_row_ptr + col_offsets
    _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs,
        col_offsets, n_cols, is_bf16)
    _softmax_grad_core(output_ptrs + n_cols, d_output_ptrs + n_cols, 
        d_input_ptrs + n_cols, col_offsets, n_cols, is_bf16)

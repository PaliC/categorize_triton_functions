import triton
import triton.language as tl
import torch

@triton.jit
def swiglu_forward_optimized(e_ptr, g_ptr, output_ptr, sigmoid_ptr, f_ptr,
    e_stride, g_stride, output_stride, sigmoid_stride, f_stride, BLOCK_SIZE:
    'tl.constexpr', n_cols):
    row_idx = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    mask = col_offset < n_cols
    e_ptr += row_idx * e_stride
    g_ptr += row_idx * g_stride
    output_ptr += row_idx * output_stride
    sigmoid_ptr += row_idx * sigmoid_stride
    f_ptr += row_idx * f_stride
    e_row = tl.load(e_ptr + col_offset, mask=mask)
    g_row = tl.load(g_ptr + col_offset, mask=mask)
    sigmoid_e_row = tl.sigmoid(e_row)
    f_row = e_row * sigmoid_e_row
    tl.store(sigmoid_ptr + col_offset, sigmoid_e_row, mask=mask)
    tl.store(f_ptr + col_offset, f_row, mask=mask)
    output_row = f_row * g_row
    tl.store(output_ptr + col_offset, output_row, mask=mask)

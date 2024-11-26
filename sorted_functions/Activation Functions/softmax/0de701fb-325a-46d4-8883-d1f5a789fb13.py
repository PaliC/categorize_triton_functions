import triton
import triton.language as tl
import torch

@triton.jit
def _softmax_kernel_fwd(output_ptr, output_row_stride, input_ptr,
    input_row_stride, n_cols, block_size: 'tl.constexpr'):
    row_index = tl.program_id(0)
    input_row_ptr = input_ptr + row_index * input_row_stride
    col_offsets = tl.arange(0, block_size)
    input_ptrs = input_row_ptr + col_offsets
    rw_mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=rw_mask, other=float('-inf'))
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denom = tl.sum(numerator, axis=0)
    sm_out = numerator / denom
    out_row_ptr = output_ptr + row_index * output_row_stride
    out_row_ptrs = out_row_ptr + col_offsets
    tl.store(out_row_ptrs, sm_out, mask=rw_mask)

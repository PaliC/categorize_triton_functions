import triton
import triton.language as tl
import torch

@triton.jit
def _softmax_kernel_bwd(output_ptr, stride_output_row, grad_ptr,
    stride_grad_row, input_ptr, stride_input_row, n_cols, block_size:
    'tl.constexpr'):
    row_index = tl.program_id(0)
    input_row_ptr = input_ptr + row_index * stride_input_row
    grad_row_ptr = grad_ptr + row_index * stride_grad_row
    col_offsets = tl.arange(0, block_size)
    rw_mask = col_offsets < n_cols
    input_row_ptrs = input_row_ptr + col_offsets
    grad_row_ptrs = grad_row_ptr + col_offsets
    probs_row = tl.load(input_row_ptrs, mask=rw_mask, other=0)
    grads_row = tl.load(grad_row_ptrs, mask=rw_mask, other=0)
    dx = probs_row * grads_row
    dsm_out = dx - probs_row * tl.sum(dx, axis=0)
    output_row_ptr = output_ptr + row_index * stride_output_row
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, dsm_out, mask=rw_mask)

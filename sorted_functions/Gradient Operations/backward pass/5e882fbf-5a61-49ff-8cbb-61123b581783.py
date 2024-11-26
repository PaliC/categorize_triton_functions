import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel_backward(output_ptr, input_ptr, grad_ptr,
    grad_row_stride, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    probs_row = tl.load(input_ptrs, mask=mask, other=0.0)
    grad_row = tl.load(grad_ptrs, mask=mask, other=0.0)
    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis=0)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask=mask)

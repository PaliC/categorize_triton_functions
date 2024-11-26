import triton
import triton.language as tl
import torch

@triton.jit
def swiglu_backward(grad_output_ptr, grad_e_ptr, grad_g_ptr, e_ptr, g_ptr,
    n_cols, sigmoid_ptr, f_ptr, grad_output_stride, grad_e_stride,
    grad_g_stride, e_stride, g_stride, sigmoid_stride, f_stride, BLOCK_SIZE:
    'tl.constexpr'):
    pid = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    mask = col_offset < n_cols
    grad_output_row = tl.load(grad_output_ptr + pid * grad_output_stride +
        col_offset, mask=mask)
    e_row = tl.load(e_ptr + pid * e_stride + col_offset, mask=mask)
    g_row = tl.load(g_ptr + pid * g_stride + col_offset, mask=mask)
    sigmoid_row = tl.load(sigmoid_ptr + pid * sigmoid_stride + col_offset,
        mask=mask)
    f_row = tl.load(f_ptr + pid * f_stride + col_offset, mask=mask)
    grad_g_row = grad_output_row * f_row
    grad_e_row = grad_output_row * g_row * sigmoid_row * (1.0 + e_row * (
        1.0 - sigmoid_row))
    tl.store(grad_e_ptr + pid * grad_e_stride + col_offset, grad_e_row,
        mask=mask)
    tl.store(grad_g_ptr + pid * grad_g_stride + col_offset, grad_g_row,
        mask=mask)

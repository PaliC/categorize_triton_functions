import triton
import triton.language as tl
import torch

@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0).cast(tl.int64)
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)

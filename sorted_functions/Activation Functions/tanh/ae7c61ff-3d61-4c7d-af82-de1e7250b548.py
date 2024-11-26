import triton
import triton.language as tl
import torch

@triton.jit
def _geglu_tanh_backward_kernel(dc, a, b, stride, n_cols: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0).cast(tl.int64)
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    db_row = dc_row * geglu_a
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 
        0.044715 * a_row * a_row))
    da_row = dc_row * b_row * (term1 + term2)
    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)

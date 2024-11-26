import triton
import triton.language as tl
import torch

@triton.jit
def _exact_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row
    h_row = f_row * g_row
    tl.store(h + offsets, h_row, mask=mask)

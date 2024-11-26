import triton
import triton.language as tl
import torch

@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: 'tl.constexpr'):
    block_id = tl.program_id(0)
    off_x = block_id * B0 + tl.arange(0, B0)
    mask = off_x < N0
    x = tl.load(x_ptr + off_x, mask=mask)
    x = x + 10.0
    tl.store(z_ptr + off_x, x, mask=mask)
    return

import triton
import triton.language as tl
import torch

@triton.jit
def is_in_bounds(x, y, z, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    in_bounds = (tl.abs(x) <= 1) * (tl.abs(y) <= 1) * (tl.abs(z) <= 1)
    if C == 1:
        in_bounds_mask = tl.view(in_bounds, (BLOCK_SIZE,))
    else:
        in_bounds_mask = tl.broadcast_to(in_bounds[:, None], (BLOCK_SIZE, C))
    return in_bounds_mask

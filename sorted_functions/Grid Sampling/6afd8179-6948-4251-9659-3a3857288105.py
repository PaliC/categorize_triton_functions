import triton
import triton.language as tl
import torch

@triton.jit
def _is_in_bounds(x, y, z, W: 'tl.constexpr', H: 'tl.constexpr', D:
    'tl.constexpr', C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    ix = (x + 1) / 2 * W - 0.5
    iy = (y + 1) / 2 * H - 0.5
    iz = (z + 1) / 2 * D - 0.5
    in_bounds = (iy >= 0) * (iy < H) * (ix >= 0) * (ix < W) * (iz >= 0) * (iz <
        D)
    in_bounds_mask = tl.broadcast_to(in_bounds[:, None], (BLOCK_SIZE, C))
    return in_bounds_mask

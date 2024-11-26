import triton
import triton.language as tl
import torch

@triton.jit
def _get_plane_grid_sample_info(gi, ix_in, iy_in, IH, IW, feature_grid_size,
    C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    BS = tl.load(feature_grid_size + offs + 0)
    grid_numel = BS * IH * IW * C
    grid_numel = tl.sum(grid_numel, axis=0) // BLOCK_SIZE
    ix11 = (ix_in + 1) / 2 * IW - 0.5
    iy11 = (iy_in + 1) / 2 * IH - 0.5
    ix = ix11 * (IW > 1)
    iy = iy11 * (IH > 1)
    ix0 = _floor(ix)
    iy0 = _floor(iy)
    return ix, iy, ix0, iy0, grid_numel

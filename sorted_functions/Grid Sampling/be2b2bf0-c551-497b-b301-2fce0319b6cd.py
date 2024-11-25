import triton
import triton.language as tl
import torch

@triton.jit
def voxel_grid_sample_one_nearest(gi, feature_grid, feature_grid_size,
    batch_index, ix_in, iy_in, iz_in, C: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    ID = tl.load(feature_grid_size + offs + 1)
    IH = tl.load(feature_grid_size + offs + 2)
    IW = tl.load(feature_grid_size + offs + 3)
    ix11 = (ix_in + 1) / 2 * IW - 0.5
    iy11 = (iy_in + 1) / 2 * IH - 0.5
    iz11 = (iz_in + 1) / 2 * ID - 0.5
    ix = ix11 * (ID > 1)
    iy = iy11 * (IH > 1)
    iz = iz11 * (IW > 1)
    unit_weight = ix * 0.0 + 1.0
    ix = _round(ix)
    iy = _round(iy)
    iz = _round(iz)
    sampled = _sample_3d(feature_grid, unit_weight, batch_index, ix, iy, iz,
        ID, IH, IW, C, BLOCK_SIZE)
    if mask_out_of_bounds_samples:
        in_bounds_mask = is_in_bounds(ix_in, iy_in, iz_in, C, BLOCK_SIZE)
        sampled *= in_bounds_mask
    return sampled

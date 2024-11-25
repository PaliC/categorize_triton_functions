import triton
import triton.language as tl
import torch

@triton.jit
def sample_grid_rep(feature_grid, feature_grid_sizes, grid_idx, sample_x,
    sample_y, sample_z, C: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    vec = _voxel_grid_sample(feature_grid, feature_grid_sizes, grid_idx,
        sample_x, sample_y, sample_z, C, NUM_GRIDS, BLOCK_SIZE,
        mask_out_of_bounds_samples)
    return vec

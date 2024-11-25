import triton
import triton.language as tl
import torch

@triton.jit
def splat_grid_rep(feature_grid, grad_image, feature_grid_sizes, grid_idx,
    sample_x, sample_y, sample_z, C: 'tl.constexpr', NUM_GRIDS:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples:
    'tl.constexpr'):
    _voxel_grid_splat(feature_grid, grad_image, feature_grid_sizes,
        grid_idx, sample_x, sample_y, sample_z, C, NUM_GRIDS, BLOCK_SIZE,
        mask_out_of_bounds_samples)

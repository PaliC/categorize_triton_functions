import triton
import triton.language as tl
import torch

@triton.jit
def _splat_grid_rep(to_splat, xy, yz, zx, batch_index, sample_x, sample_y,
    sample_z, batch_size: 'tl.constexpr', C: 'tl.constexpr', D:
    'tl.constexpr', H: 'tl.constexpr', W: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr', shape_representation: 'tl.constexpr'):
    if shape_representation == 0:
        _grid_splat(to_splat, xy, batch_index, sample_x, sample_y,
            batch_size, C, H, W, BLOCK_SIZE)
        _grid_splat(to_splat, yz, batch_index, sample_y, sample_z,
            batch_size, C, D, H, BLOCK_SIZE)
        _grid_splat(to_splat, zx, batch_index, sample_z, sample_x,
            batch_size, C, W, D, BLOCK_SIZE)
    else:
        _voxel_grid_splat(to_splat, xy, batch_index, sample_x, sample_y,
            sample_z, batch_size, C, D, H, W, BLOCK_SIZE)

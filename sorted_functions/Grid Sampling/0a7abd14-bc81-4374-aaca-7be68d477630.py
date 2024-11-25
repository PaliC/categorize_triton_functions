import triton
import triton.language as tl
import torch

@triton.jit
def _sample_grid_rep(xy, yz, zx, batch_index, sample_x, sample_y, sample_z,
    batch_size: 'tl.constexpr', C: 'tl.constexpr', D: 'tl.constexpr', H:
    'tl.constexpr', W: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr',
    shape_representation: 'tl.constexpr'):
    if shape_representation == 0:
        a = _grid_sample(xy, batch_index, sample_x, sample_y, batch_size, C,
            H, W, BLOCK_SIZE)
        b = _grid_sample(yz, batch_index, sample_y, sample_z, batch_size, C,
            D, H, BLOCK_SIZE)
        c = _grid_sample(zx, batch_index, sample_z, sample_x, batch_size, C,
            W, D, BLOCK_SIZE)
        vec = a + b + c
    else:
        vec = _voxel_grid_sample(xy, batch_index, sample_x, sample_y,
            sample_z, batch_size, C, D, H, W, BLOCK_SIZE)
    vec = tl.view(vec, (BLOCK_SIZE, C))
    return vec

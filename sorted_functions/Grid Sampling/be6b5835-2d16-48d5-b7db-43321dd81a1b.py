import triton
import triton.language as tl
import torch

@triton.jit
def _voxel_grid_splat_one(gi, to_splat, grad_feature_grid,
    feature_grid_size, batch_index, ix_in, iy_in, iz_in, IH, IW, ID, C:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples:
    'tl.constexpr'):
    ix, iy, iz, ix0, iy0, iz0, grid_numel = _get_voxel_grid_sample_info(gi,
        ix_in, iy_in, iz_in, ID, IH, IW, feature_grid_size, C, BLOCK_SIZE)
    (V000x, V000y, V000z, V100x, V100y, V100z, V010x, V010y, V010z, V001x,
        V001y, V001z, V101x, V101y, V101z, V011x, V011y, V011z, V110x,
        V110y, V110z, V111x, V111y, V111z, x, y, z
        ) = _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * (1 - y) * (1 - z),
        batch_index, V000x, V000y, V000z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * (1 - y) * z,
        batch_index, V100x, V100y, V100z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * y * (1 - z),
        batch_index, V010x, V010y, V010z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * (1 - y) * (1 - z),
        batch_index, V001x, V001y, V001z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * (1 - y) * z, batch_index,
        V101x, V101y, V101z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * y * (1 - z), batch_index,
        V011x, V011y, V011z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * y * z, batch_index,
        V110x, V110y, V110z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * y * z, batch_index, V111x,
        V111y, V111z, ID, IH, IW, C, BLOCK_SIZE)
    return grid_numel

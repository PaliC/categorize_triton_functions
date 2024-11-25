import triton
import triton.language as tl
import torch

@triton.jit
def _plane_grid_sample_one(gi, feature_grid, feature_grid_size, batch_index,
    ix_in, iy_in, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr',
    mask_out_of_bounds_samples: 'tl.constexpr'):
    ix, iy, ix0, iy0, grid_numel = _get_plane_grid_sample_info(gi, ix_in,
        iy_in, IH, IW, feature_grid_size, C, BLOCK_SIZE)
    V00x, V00y, V10x, V10y, V01x, V01y, V11x, V11y, x, y = (
        _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0))
    sampled = _sample_2d(feature_grid, (1 - x) * (1 - y), batch_index, V00x,
        V00y, IH, IW, C, BLOCK_SIZE) + _sample_2d(feature_grid, x * (1 - y),
        batch_index, V01x, V01y, IH, IW, C, BLOCK_SIZE) + _sample_2d(
        feature_grid, (1 - x) * y, batch_index, V10x, V10y, IH, IW, C,
        BLOCK_SIZE) + _sample_2d(feature_grid, x * y, batch_index, V11x,
        V11y, IH, IW, C, BLOCK_SIZE)
    return sampled, grid_numel

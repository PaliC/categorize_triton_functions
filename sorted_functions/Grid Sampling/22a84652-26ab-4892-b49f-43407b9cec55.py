import triton
import triton.language as tl
import torch

@triton.jit
def _voxel_grid_splat(to_splat, grad_feature_grid, feature_grid_size,
    batch_index, ix_in, iy_in, iz_in, C: 'tl.constexpr', NUM_GRIDS:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples:
    'tl.constexpr'):
    feature_grid_offs = tl.zeros((1,), dtype=tl.int32)
    for gi in range(NUM_GRIDS):
        offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        ID = tl.load(feature_grid_size + offs + 1)
        IH = tl.load(feature_grid_size + offs + 2)
        IW = tl.load(feature_grid_size + offs + 3)
        ID_ = tl.sum(ID, axis=0) // BLOCK_SIZE
        IH_ = tl.sum(IH, axis=0) // BLOCK_SIZE
        IW_ = tl.sum(IW, axis=0) // BLOCK_SIZE
        voxel_grid = (ID_ - 1) * (IH_ - 1) * (IW_ - 1)
        if mask_out_of_bounds_samples:
            in_bounds_mask = is_in_bounds(ix_in, iy_in, iz_in, C, BLOCK_SIZE)
            if C == 1:
                in_bounds_mask = in_bounds_mask[:, None]
            to_splat = to_splat * in_bounds_mask
        else:
            to_splat = to_splat
        if voxel_grid > 0:
            grid_numel = _voxel_grid_splat_one(gi, to_splat, 
                grad_feature_grid + feature_grid_offs, feature_grid_size,
                batch_index, ix_in, iy_in, iz_in, IH, IW, ID, C, BLOCK_SIZE,
                mask_out_of_bounds_samples)
        elif ID_ == 1:
            grid_numel = _plane_grid_splat_one(gi, to_splat, 
                grad_feature_grid + feature_grid_offs, feature_grid_size,
                batch_index, ix_in, iy_in, IH, IW, C, BLOCK_SIZE,
                mask_out_of_bounds_samples)
        elif IH_ == 1:
            grid_numel = _plane_grid_splat_one(gi, to_splat, 
                grad_feature_grid + feature_grid_offs, feature_grid_size,
                batch_index, ix_in, iz_in, ID, IW, C, BLOCK_SIZE,
                mask_out_of_bounds_samples)
        else:
            grid_numel = _plane_grid_splat_one(gi, to_splat, 
                grad_feature_grid + feature_grid_offs, feature_grid_size,
                batch_index, iy_in, iz_in, ID, IH, C, BLOCK_SIZE,
                mask_out_of_bounds_samples)
        feature_grid_offs += grid_numel

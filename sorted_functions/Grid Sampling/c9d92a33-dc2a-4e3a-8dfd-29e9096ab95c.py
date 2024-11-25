import triton
import triton.language as tl
import torch

@triton.jit
def fwbw_splatter_init(directions, origins, grid_idx, near, far,
    splatting_feature, mask, num_samples: 'tl.constexpr', num_samples_inf:
    'tl.constexpr', num_rays: 'tl.constexpr', grid_channel: 'tl.constexpr',
    feature_channel: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    tot_num_samples = num_samples + num_samples_inf
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = offs < num_rays
    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1
    offs_features = (pid * BLOCK_SIZE * feature_channel + feature_channel *
        tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, feature_channel)[
        None, :])
    offs_features_mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None
        ] < num_rays
    center_x = tl.load(origins + offs_x, mask=offs_x < num_rays * 3)
    center_y = tl.load(origins + offs_y, mask=offs_y < num_rays * 3)
    center_z = tl.load(origins + offs_z, mask=offs_z < num_rays * 3)
    ray_x = tl.load(directions + offs_x, mask=offs_x < num_rays * 3)
    ray_y = tl.load(directions + offs_y, mask=offs_y < num_rays * 3)
    ray_z = tl.load(directions + offs_z, mask=offs_z < num_rays * 3)
    near_buffer = tl.load(near + offs, mask=offs_mask)
    far_buffer = tl.load(far + offs, mask=offs_mask)
    grid_idx_buffer = tl.load(grid_idx + offs, mask=offs_mask)
    sample_index_buffer = tl.arange(0, BLOCK_SIZE
        ) * tot_num_samples + pid * BLOCK_SIZE * tot_num_samples + 1
    feature = tl.load(splatting_feature + offs_features, mask=
        offs_features_mask)
    mask = tl.load(mask + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:,
        None], mask=offs_features_mask)
    mask = tl.view(mask, (BLOCK_SIZE, 1))
    return (tot_num_samples, pid, offs, offs_mask, offs_features,
        offs_features_mask, center_x, center_y, center_z, ray_x, ray_y,
        ray_z, near_buffer, far_buffer, grid_idx_buffer,
        sample_index_buffer, feature, mask)

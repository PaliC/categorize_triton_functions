import triton
import triton.language as tl
import torch

@triton.jit
def fw_kernel_wMLP(feature_grid, feature_grid_sizes, input_feature_grid,
    input_feature_grid_sizes, directions, origins, grid_idx, near, far,
    splatting_feature, mask, mlp_params, DIM_HIDDEN: 'tl.constexpr', DIM_IN:
    'tl.constexpr', DIM_OUT: 'tl.constexpr', num_samples: 'tl.constexpr',
    num_samples_inf: 'tl.constexpr', num_rays: 'tl.constexpr', grid_channel:
    'tl.constexpr', NUM_GRIDS: 'tl.constexpr', feature_channel:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples:
    'tl.constexpr', contract_coords: 'tl.constexpr', disparity_at_inf:
    'tl.constexpr'):
    (tot_num_samples, pid, offs, offs_mask, offs_features,
        offs_features_mask, center_x, center_y, center_z, ray_x, ray_y,
        ray_z, near_buffer, far_buffer, grid_idx_buffer,
        sample_index_buffer, feature_buffer, mask_buffer) = (fwbw_splatter_init
        (directions, origins, grid_idx, near, far, splatting_feature, mask,
        num_samples, num_samples_inf, num_rays, grid_channel,
        feature_channel, BLOCK_SIZE))
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = depth_inv_sphere(far_buffer, disparity_at_inf,
                num_samples_inf, step - num_samples)
        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z
        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y,
                sample_z)
        prev_vec = sample_grid_rep(input_feature_grid,
            input_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y,
            sample_z, feature_channel, NUM_GRIDS, BLOCK_SIZE,
            mask_out_of_bounds_samples)
        fused_feature = feature_buffer + prev_vec
        fused_feature = fused_feature * mask_buffer
        splat_grid_rep(fused_feature, feature_grid, feature_grid_sizes,
            grid_idx_buffer, sample_x, sample_y, sample_z, grid_channel,
            NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)

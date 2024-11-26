import triton
import triton.language as tl
import torch

@triton.jit
def bw_kernel_wMLP(grad_feature_grid, grad_feature_grid_sizes, feature_grid,
    feature_grid_sizes, input_feature_grid, input_feature_grid_sizes,
    directions, origins, grid_idx, near, far, splatting_feature, mask,
    mlp_params, DIM_HIDDEN: 'tl.constexpr', DIM_IN: 'tl.constexpr', DIM_OUT:
    'tl.constexpr', num_samples: 'tl.constexpr', num_samples_inf:
    'tl.constexpr', num_rays: 'tl.constexpr', grid_channel: 'tl.constexpr',
    NUM_GRIDS: 'tl.constexpr', feature_channel: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr',
    contract_coords: 'tl.constexpr', disparity_at_inf: 'tl.constexpr',
    grad_splatting_feature, grad_mlp_params, grad_input_feature_grid):
    (tot_num_samples, pid, offs, offs_mask, offs_features,
        offs_features_mask, center_x, center_y, center_z, ray_x, ray_y,
        ray_z, near_buffer, far_buffer, grid_idx_buffer,
        sample_index_buffer, feature_buffer, mask_buffer) = (fwbw_splatter_init
        (directions, origins, grid_idx, near, far, splatting_feature, mask,
        num_samples, num_samples_inf, num_rays, grid_channel,
        feature_channel, BLOCK_SIZE))
    depth = near_buffer
    grad_splatting_feature_buffer = tl.zeros((BLOCK_SIZE, feature_channel),
        dtype=tl.float32)
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
        grad_vec = sample_grid_rep(grad_feature_grid,
            grad_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y,
            sample_z, grid_channel, NUM_GRIDS, BLOCK_SIZE,
            mask_out_of_bounds_samples)
        grad_vec = grad_vec * mask_buffer
        fused_feature = feature_buffer + prev_vec
        splat_grid_rep(grad_splatting, grad_input_feature_grid,
            input_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y,
            sample_z, feature_channel, NUM_GRIDS, BLOCK_SIZE,
            mask_out_of_bounds_samples)
        grad_splatting_feature_buffer += grad_splatting
    tl.store(grad_splatting_feature + offs_features,
        grad_splatting_feature_buffer, mask=offs_features_mask)
    update_mlp_params(grad_mlp_params, DIM_IN, DIM_HIDDEN)

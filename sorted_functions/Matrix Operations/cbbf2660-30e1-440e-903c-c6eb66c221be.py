import triton
import triton.language as tl
import torch

@triton.jit
def _fw_kernel(xy, yz, zx, xy_color, yz_color, zx_color, rays, centers,
    weights, biases, weight_opacity, bias_opacity, weight_color, bias_color,
    rays_encoding, negative_log_transmittance, expected_depth,
    expected_features, near, far, effective_num_samples, num_samples:
    'tl.constexpr', num_samples_inf: 'tl.constexpr', gain: 'tl.constexpr',
    batch_size: 'tl.constexpr', num_rays_per_batch: 'tl.constexpr', C:
    'tl.constexpr', OUT_C: 'tl.constexpr', H: 'tl.constexpr', W:
    'tl.constexpr', D: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr',
    transmittance_thr: 'tl.constexpr', mask_out_of_bounds_samples:
    'tl.constexpr', inject_noise: 'tl.constexpr', inject_noise_sigma:
    'tl.constexpr', inject_noise_seed, contract_coords: 'tl.constexpr',
    contract_perc_foreground: 'tl.constexpr', disparity_at_inf:
    'tl.constexpr', shape_representation: 'tl.constexpr', activation_fun:
    'tl.constexpr', use_separate_color_rep: 'tl.constexpr'):
    tot_num_samples = num_samples + num_samples_inf
    num_rays = num_rays_per_batch * batch_size
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1
    offs_features = pid * BLOCK_SIZE * OUT_C + OUT_C * tl.arange(0, BLOCK_SIZE
        )[:, None] + tl.arange(0, OUT_C)[None, :]
    offs_features_mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None
        ] < num_rays
    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays * 3)
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays * 3)
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays * 3)
    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays * 3)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays * 3)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays * 3)
    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ) // num_rays_per_batch
    near_buffer = tl.load(near + offs, mask=offs < num_rays)
    far_buffer = tl.load(far + offs, mask=offs < num_rays)
    effective_num_samples_buffer = tl.zeros((1,), dtype=tl.int32)
    depth = near_buffer
    seed_buffer = tl.load(inject_noise_seed + offs, mask=offs < num_rays)
    sample_index_buffer = tl.arange(0, BLOCK_SIZE
        ) * tot_num_samples + pid * BLOCK_SIZE * tot_num_samples + 1
    expected_depth_buffer = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    expected_features_buffer = tl.zeros((BLOCK_SIZE, OUT_C), dtype=tl.float32)
    prev_transmittance = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    negative_log_transmittance_buffer = tl.zeros((BLOCK_SIZE,), dtype=tl.
        float32)
    w1, w2, wr, wo, wc, b1, b2, br, bo, bc, w2c, b2c = _load_mlp_weights(
        weights, biases, weight_opacity, bias_opacity, weight_color,
        bias_color, C, OUT_C, BLOCK_SIZE)
    rays_encoding_buffer = tl.load(rays_encoding + pid * BLOCK_SIZE * C + C *
        tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, C)[None, :])
    transmittance = tl.exp(-negative_log_transmittance_buffer)
    zero_value = tl.zeros((BLOCK_SIZE,), tl.float32)
    zero_color = tl.zeros((BLOCK_SIZE, OUT_C), tl.float32)
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = _depth_lin(near_buffer, far_buffer, num_samples, step)
            depth_prev = _depth_lin(near_buffer, far_buffer, num_samples, 
                step - 1)
        else:
            depth = _depth_inv_sphere(far_buffer, disparity_at_inf,
                num_samples_inf, step - num_samples)
            depth_prev = _depth_inv_sphere(far_buffer, disparity_at_inf,
                num_samples_inf, step - num_samples - 1)
        delta = depth - depth_prev
        if tl.sum(transmittance > transmittance_thr, axis=0):
            sample_x = center_x + depth * ray_x
            sample_y = center_y + depth * ray_y
            sample_z = center_z + depth * ray_z
            if contract_coords:
                sample_x, sample_y, sample_z = _contract_pi(sample_x,
                    sample_y, sample_z, contract_perc_foreground)
            vec = _sample_grid_rep(xy, yz, zx, batch_index, sample_x,
                sample_y, sample_z, batch_size, C, D, H, W, BLOCK_SIZE,
                shape_representation)
            if mask_out_of_bounds_samples:
                in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z,
                    W, H, D, C, BLOCK_SIZE)
                vec = vec * in_bounds_mask
            vec = tl.maximum(tl.dot(vec, w1, allow_tf32=ALLOW_TF32) + b1, 0.0)
            vec = tl.maximum(tl.dot(vec, w2, allow_tf32=ALLOW_TF32) + b2, 0.0)
            value = tl.view(tl.sum(wo * vec, axis=1), (BLOCK_SIZE,)) + bo
            if inject_noise:
                r = _int_to_randn(sample_index_buffer, sample_index_buffer +
                    num_rays * tot_num_samples, seed_buffer)
                value = value + r * inject_noise_sigma
            if activation_fun == 0:
                value_act = _softplus(value)
            else:
                value_act = tl.maximum(value, 0.0)
            value = delta * gain * value_act
            if use_separate_color_rep:
                vec_color = _sample_grid_rep(xy_color, yz_color, zx_color,
                    batch_index, sample_x, sample_y, sample_z, batch_size,
                    C, D, H, W, BLOCK_SIZE, shape_representation)
                vec_color = vec_color + rays_encoding_buffer
                if mask_out_of_bounds_samples:
                    in_bounds_mask = _is_in_bounds(sample_x, sample_y,
                        sample_z, W, H, D, C, BLOCK_SIZE)
                    vec_color = vec_color * in_bounds_mask
                vec_color1 = tl.maximum(tl.dot(vec_color, wr, allow_tf32=
                    ALLOW_TF32) + br, 0.0)
                vec_color2 = tl.maximum(tl.dot(vec_color1, w2c, allow_tf32=
                    ALLOW_TF32) + b2c, 0.0)
                log_color = tl.dot(vec_color2, wc, allow_tf32=ALLOW_TF32) + bc
            else:
                vecr = tl.maximum(tl.dot(vec, wr, allow_tf32=ALLOW_TF32) +
                    br + rays_encoding_buffer, 0.0)
                log_color = tl.dot(vecr, wc, allow_tf32=ALLOW_TF32) + bc
            color = _color_activation(log_color)
            effective_ns_increment = 1
        else:
            value = zero_value
            color = zero_color
            effective_ns_increment = 0
        negative_log_transmittance_buffer = (
            negative_log_transmittance_buffer + value)
        transmittance = tl.exp(-negative_log_transmittance_buffer)
        render_weights = prev_transmittance - transmittance
        expected_depth_buffer = expected_depth_buffer + render_weights * depth
        render_weights_bcast = tl.broadcast_to(prev_transmittance[:, None],
            (BLOCK_SIZE, OUT_C)) - tl.broadcast_to(transmittance[:, None],
            (BLOCK_SIZE, OUT_C))
        feature_render = color * render_weights_bcast
        expected_features_buffer = expected_features_buffer + feature_render
        prev_transmittance = transmittance
        sample_index_buffer = sample_index_buffer + 1
        effective_num_samples_buffer = (effective_num_samples_buffer +
            effective_ns_increment)
    tl.store(negative_log_transmittance + offs,
        negative_log_transmittance_buffer, mask=offs < num_rays)
    tl.store(expected_depth + offs, expected_depth_buffer, mask=offs < num_rays
        )
    tl.store(effective_num_samples + pid + tl.arange(0, 1),
        effective_num_samples_buffer)
    tl.store(expected_features + offs_features, expected_features_buffer,
        mask=offs_features_mask)

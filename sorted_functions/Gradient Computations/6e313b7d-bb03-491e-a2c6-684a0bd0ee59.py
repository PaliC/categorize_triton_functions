import triton
import triton.language as tl
import torch

@triton.jit
def _bw_kernel(xy, yz, zx, xy_color, yz_color, zx_color, rays, centers,
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
    'tl.constexpr', use_separate_color_rep: 'tl.constexpr',
    grad_negative_log_transmittance, grad_expected_depth,
    grad_expected_features, grad_xy, grad_yz, grad_zx, grad_xy_color,
    grad_yz_color, grad_zx_color, grad_weights, grad_biases,
    grad_weight_opacity, grad_bias_opacity, grad_weight_color,
    grad_bias_color, grad_rays_enc):
    tot_num_samples = num_samples + num_samples_inf
    num_rays = num_rays_per_batch * batch_size
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = offs < num_rays
    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1
    offs_features = pid * BLOCK_SIZE * OUT_C + OUT_C * tl.arange(0, BLOCK_SIZE
        )[:, None] + tl.arange(0, OUT_C)[None, :]
    offs_features_mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None
        ] < num_rays
    offs_CC = tl.arange(0, C)[:, None] * C + tl.arange(0, C)[None, :]
    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays * 3)
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays * 3)
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays * 3)
    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays * 3)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays * 3)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays * 3)
    rays_enc_offs = pid * BLOCK_SIZE * C + C * tl.arange(0, BLOCK_SIZE)[:, None
        ] + tl.arange(0, C)[None, :]
    rays_enc_mask = rays_enc_offs < num_rays * C
    rays_encoding_buffer = tl.load(rays_encoding + rays_enc_offs, mask=
        rays_enc_mask)
    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ) // num_rays_per_batch
    near_buffer = tl.load(near + offs, mask=offs < num_rays)
    far_buffer = tl.load(far + offs, mask=offs < num_rays)
    seed_buffer = tl.load(inject_noise_seed + offs, mask=offs < num_rays)
    sample_index_buffer = (tl.arange(0, BLOCK_SIZE) * tot_num_samples + pid *
        BLOCK_SIZE * tot_num_samples + 1 + tot_num_samples - 1)
    depth = far_buffer
    grad_negative_log_transmittance_buffer = tl.load(
        grad_negative_log_transmittance + offs, mask=offs_mask, other=0.0)
    grad_expected_features_buffer = tl.load(grad_expected_features +
        offs_features, mask=offs_features_mask, other=0.0)
    grad_expected_depth_buffer = tl.load(grad_expected_depth + offs, mask=
        offs_mask, other=0.0)
    prev_proj_depth = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    prev_proj_features = tl.zeros((BLOCK_SIZE, OUT_C), dtype=tl.float32)
    negative_log_transmittance_buffer = tl.load(negative_log_transmittance +
        offs, mask=offs_mask, other=0.0)
    transmittance = tl.exp(-negative_log_transmittance_buffer)
    prev_grad_opacity = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    d_w2 = tl.zeros((C, C), dtype=tl.float32)
    d_w1 = tl.zeros((C, C), dtype=tl.float32)
    d_b1 = tl.zeros((C,), dtype=tl.float32)
    d_b2 = tl.zeros((C,), dtype=tl.float32)
    d_w2c = tl.zeros((C, C), dtype=tl.float32)
    d_b2c = tl.zeros((C,), dtype=tl.float32)
    d_wr = tl.zeros((C, C), dtype=tl.float32)
    d_br = tl.zeros((C,), dtype=tl.float32)
    d_wo = tl.zeros((C,), dtype=tl.float32)
    d_bo = tl.zeros((1,), dtype=tl.float32)
    d_wc = tl.zeros((C, OUT_C), dtype=tl.float32)
    d_wc = tl.zeros((OUT_C, C), dtype=tl.float32)
    d_bc = tl.zeros((OUT_C,), dtype=tl.float32)
    d_rays_enc = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)
    zero_w = tl.zeros((C, C), dtype=tl.float32)
    zero_b = tl.zeros((C,), dtype=tl.float32)
    zero_vec = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)
    w1, w2, wr, wo, wc, b1, b2, br, bo, bc, w2c, b2c = _load_mlp_weights(
        weights, biases, weight_opacity, bias_opacity, weight_color,
        bias_color, C, OUT_C, BLOCK_SIZE)
    prev_transmittance = transmittance
    for step in range(tot_num_samples):
        if step < num_samples_inf:
            depth = _depth_inv_sphere(far_buffer, disparity_at_inf,
                num_samples_inf, num_samples_inf - step - 1)
            depth_prev = _depth_inv_sphere(far_buffer, disparity_at_inf,
                num_samples_inf, num_samples_inf - step - 2)
        else:
            depth = _depth_lin(near_buffer, far_buffer, num_samples, 
                num_samples - (step - num_samples_inf) - 1)
            depth_prev = _depth_lin(near_buffer, far_buffer, num_samples, 
                num_samples - (step - num_samples_inf) - 2)
        delta = depth - depth_prev
        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z
        if contract_coords:
            sample_x, sample_y, sample_z = _contract_pi(sample_x, sample_y,
                sample_z, contract_perc_foreground)
        vec = _sample_grid_rep(xy, yz, zx, batch_index, sample_x, sample_y,
            sample_z, batch_size, C, D, H, W, BLOCK_SIZE, shape_representation)
        if mask_out_of_bounds_samples:
            in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W,
                H, D, C, BLOCK_SIZE)
            vec = vec * in_bounds_mask
        vec1 = tl.maximum(tl.dot(vec, w1, allow_tf32=ALLOW_TF32) + b1, 0.0)
        vec2 = tl.maximum(tl.dot(vec1, w2, allow_tf32=ALLOW_TF32) + b2, 0.0)
        value = tl.sum(vec2 * wo, axis=1) + bo
        if inject_noise:
            r = _int_to_randn(sample_index_buffer, sample_index_buffer + 
                num_rays * tot_num_samples, seed_buffer)
            value = value + r * inject_noise_sigma
        if activation_fun == 0:
            value_act = _softplus(value)
        else:
            value_act = tl.maximum(value, 0.0)
        delta_value = gain * value_act * delta
        if use_separate_color_rep:
            vec_color = _sample_grid_rep(xy_color, yz_color, zx_color,
                batch_index, sample_x, sample_y, sample_z, batch_size, C, D,
                H, W, BLOCK_SIZE, shape_representation)
            vec_color = vec_color + rays_encoding_buffer
            if mask_out_of_bounds_samples:
                in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z,
                    W, H, D, C, BLOCK_SIZE)
                vec_color = vec_color * in_bounds_mask
            vec_color1 = tl.maximum(tl.dot(vec_color, wr, allow_tf32=
                ALLOW_TF32) + br, 0.0)
            vecr = tl.maximum(tl.dot(vec_color1, w2c, allow_tf32=ALLOW_TF32
                ) + b2c, 0.0)
        else:
            vecr = tl.maximum(tl.dot(vec2, wr, allow_tf32=ALLOW_TF32) + br +
                rays_encoding_buffer, 0.0)
        log_color = tl.dot(vecr, wc, allow_tf32=ALLOW_TF32) + bc
        color = _color_activation(log_color)
        proj_features = color * grad_expected_features_buffer
        proj_depth = depth * grad_expected_depth_buffer
        prev_transmittance = transmittance
        opacity_grad_now = prev_transmittance * (proj_depth -
            prev_proj_depth + tl.sum(proj_features - prev_proj_features,
            axis=1))
        prev_grad_opacity = prev_grad_opacity + opacity_grad_now
        grad_value_act = delta * (prev_grad_opacity +
            grad_negative_log_transmittance_buffer)
        if activation_fun == 0:
            grad_value_act = _d_softplus(grad_value_act, value)
        else:
            grad_value_act = grad_value_act * (value > 0.0)
        grad_value = gain * grad_value_act
        grad_value = tl.expand_dims(grad_value, 1)
        d_wo_ = tl.sum(vec2 * tl.broadcast_to(grad_value, (BLOCK_SIZE, C)),
            axis=0)
        d_bo_ = tl.sum(grad_value, axis=0)
        d_vec2_1 = wo * grad_value
        negative_log_transmittance_buffer = (
            negative_log_transmittance_buffer - delta_value)
        transmittance = tl.exp(-negative_log_transmittance_buffer)
        """
        transmittance_diff = tl.broadcast_to(
            tl.view(transmittance, (BLOCK_SIZE, 1)), (BLOCK_SIZE, OUT_C)
        ) - tl.broadcast_to(
            tl.view(prev_transmittance, (BLOCK_SIZE, 1)), (BLOCK_SIZE, OUT_C)
        )  # = rendering weights for the given step
        """
        transmittance_diff = tl.broadcast_to(tl.expand_dims(transmittance, 
            1), (BLOCK_SIZE, OUT_C)) - tl.broadcast_to(tl.expand_dims(
            prev_transmittance, 1), (BLOCK_SIZE, OUT_C))
        d_color = grad_expected_features_buffer * transmittance_diff
        d_log_color = _d_color_activation(d_color, log_color)
        d_vecr, d_wc_, d_bc_ = _d_linear(d_log_color, wc, bc, vecr)
        if use_separate_color_rep:
            d_vec2_12 = tl.view(d_vec2_1, (BLOCK_SIZE, C))
        else:
            d_vec2_2, d_wr_, d_br_ = _d_linear_relu(d_vecr, wr, br, vecr, vec2)
            d_vec2_12 = tl.view(d_vec2_1, (BLOCK_SIZE, C)) + tl.view(d_vec2_2,
                (BLOCK_SIZE, C))
        d_vec1, d_w2_, d_b2_ = _d_linear_relu(d_vec2_12, w2, b2, vec2, vec1)
        d_vec, d_w1_, d_b1_ = _d_linear_relu(d_vec1, w1, b1, vec1, vec)
        if mask_out_of_bounds_samples:
            in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z, W,
                H, D, C, BLOCK_SIZE)
            d_vec = d_vec * in_bounds_mask
        _splat_grid_rep(d_vec, grad_xy, grad_yz, grad_zx, batch_index,
            sample_x, sample_y, sample_z, batch_size, C, D, H, W,
            BLOCK_SIZE, shape_representation)
        if use_separate_color_rep:
            d_vec_color1, d_w2c_, d_b2c_ = _d_linear_relu(d_vecr, w2c, b2c,
                vecr, vec_color1)
            d_vec_color, d_wr_, d_br_ = _d_linear_relu(d_vec_color1, wr, br,
                vec_color1, vec_color)
            if mask_out_of_bounds_samples:
                in_bounds_mask = _is_in_bounds(sample_x, sample_y, sample_z,
                    W, H, D, C, BLOCK_SIZE)
                d_vec_color = d_vec_color * in_bounds_mask
            d_rays_enc_ = tl.view(d_vec_color, (BLOCK_SIZE, C))
            _splat_grid_rep(d_vec_color, grad_xy_color, grad_yz_color,
                grad_zx_color, batch_index, sample_x, sample_y, sample_z,
                batch_size, C, D, H, W, BLOCK_SIZE, shape_representation)
        else:
            d_vec_color = zero_vec
            d_w2c_ = zero_w
            d_b2c_ = zero_b
            d_rays_enc_ = d_vecr * (vecr > 0.0)
        d_wc += d_wc_
        d_bc += d_bc_
        d_wr += d_wr_
        d_br += d_br_
        d_w2 += d_w2_
        d_w1 += d_w1_
        d_b1 += d_b1_
        d_b2 += d_b2_
        d_wo += d_wo_
        d_bo += d_bo_
        d_w2c += d_w2c_
        d_b2c += d_b2c_
        d_rays_enc += d_rays_enc_
        prev_proj_depth = proj_depth
        prev_proj_features = proj_features
        sample_index_buffer = sample_index_buffer - 1
    tl.atomic_add(grad_weights + offs_CC, d_w1)
    tl.atomic_add(grad_weights + C * C + offs_CC, d_w2)
    tl.atomic_add(grad_weights + 2 * C * C + offs_CC, d_wr)
    tl.atomic_add(grad_weights + 3 * C * C + offs_CC, d_w2c)
    tl.atomic_add(grad_biases + tl.arange(0, C), d_b1)
    tl.atomic_add(grad_biases + C + tl.arange(0, C), d_b2)
    tl.atomic_add(grad_biases + 2 * C + tl.arange(0, C), d_br)
    tl.atomic_add(grad_biases + 3 * C + tl.arange(0, C), d_b2c)
    tl.atomic_add(grad_weight_opacity + tl.arange(0, C), d_wo)
    tl.atomic_add(grad_bias_opacity + tl.arange(0, 1), d_bo)
    tl.atomic_add(grad_weight_color + tl.arange(0, OUT_C)[:, None] * C + tl
        .arange(0, C)[None, :], d_wc)
    tl.atomic_add(grad_bias_color + tl.arange(0, OUT_C), d_bc)
    tl.store(grad_rays_enc + rays_enc_offs, d_rays_enc, mask=rays_enc_mask)

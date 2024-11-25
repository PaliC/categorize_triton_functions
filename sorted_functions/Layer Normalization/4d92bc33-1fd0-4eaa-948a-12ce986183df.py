import triton
import triton.language as tl
import torch

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim',
    'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': lambda args: next_power_of_2(args[
    'batch_dim']), 'BLOCK_SIZE_SPATIAL': BLOCK_SIZE_SPATIAL_heuristic})
@triton.jit
def batch_norm_backward_kernel(output_grad_pointer, input_pointer,
    mean_pointer, inv_std_pointer, weight_pointer, input_grad_pointer,
    weight_grad_pointer, bias_grad_pointer, batch_dim, spatial_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    output_grad_spatial_stride, input_batch_stride, input_feat_stride,
    input_spatial_stride, input_grad_batch_stride, input_grad_feat_stride,
    input_grad_spatial_stride, affine: 'tl.constexpr', BLOCK_SIZE_BATCH:
    'tl.constexpr', BLOCK_SIZE_SPATIAL: 'tl.constexpr'):
    """
    Calculates the input gradient of batch normalization.

    Args:
        output_grad_pointer: Pointer to layer normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim, spatial_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        mean_pointer: Pointer to the input's mean.
            The mean should be of shape [feat_dim].
        inv_std_pointer: Pointer to the input's inverse standard deviation.
            The inverse standard deviation should be of shape [feat_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim].
        weight_grad_pointer: Pointer to an optional container the weights' gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        output_grad_spatial_stride: Stride necessary to jump one element along the
            output gradients' spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        affine: Flag for performing an affine transformation on the normalized output.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    feat_pid = tl.program_id(axis=0)
    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim
    mean = tl.load(feat_pid + mean_pointer)
    inv_std = tl.load(feat_pid + inv_std_pointer)
    term1 = 0.0
    term2 = 0.0
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0,
            BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_output_grad_pointer = (output_grad_pointer + 
            output_grad_feat_stride * feat_pid + output_grad_batch_stride *
            batch_offset[:, None] + output_grad_spatial_stride *
            spatial_offset[None, :])
        curr_input_pointer = (input_pointer + input_feat_stride * feat_pid +
            input_batch_stride * batch_offset[:, None] + 
            input_spatial_stride * spatial_offset[None, :])
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] &
            spatial_mask[None, :])
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer, mask=
            batch_mask[:, None] & spatial_mask[None, :])
        term1 += tl.sum(curr_pre_lin * curr_output_grad)
        term2 += tl.sum(curr_output_grad)
    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        weight_grad = 0.0
        bias_grad = 0.0
    else:
        weight = 1.0
    count = batch_dim * spatial_dim
    term1 *= weight / count
    term2 *= weight / count
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0,
            BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_output_grad_pointer = (output_grad_pointer + 
            output_grad_feat_stride * feat_pid + output_grad_batch_stride *
            batch_offset[:, None] + output_grad_spatial_stride *
            spatial_offset[None, :])
        curr_input_pointer = (input_pointer + input_feat_stride * feat_pid +
            input_batch_stride * batch_offset[:, None] + 
            input_spatial_stride * spatial_offset[None, :])
        curr_input_grad_pointer = (input_grad_pointer + 
            input_grad_feat_stride * feat_pid + input_grad_batch_stride *
            batch_offset[:, None] + input_grad_spatial_stride *
            spatial_offset[None, :])
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] &
            spatial_mask[None, :])
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer, mask=
            batch_mask[:, None] & spatial_mask[None, :])
        curr_input_grad = inv_std * (weight * curr_output_grad - (term1 *
            curr_pre_lin + term2))
        tl.store(curr_input_grad_pointer, curr_input_grad, mask=batch_mask[
            :, None] & spatial_mask[None, :])
        if affine:
            weight_grad += tl.sum(curr_pre_lin * curr_output_grad)
            bias_grad += tl.sum(curr_output_grad)
    if affine:
        tl.store(feat_pid + weight_grad_pointer, weight_grad)
        tl.store(feat_pid + bias_grad_pointer, bias_grad)

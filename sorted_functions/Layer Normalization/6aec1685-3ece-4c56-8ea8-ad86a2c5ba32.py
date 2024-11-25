import triton
import triton.language as tl
import torch

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def layer_norm_forward_kernel(input_pointer, weight_pointer, bias_pointer,
    mean_pointer, inv_std_pointer, output_pointer, batch_dim, feat_dim,
    input_batch_stride, input_feat_stride, output_batch_stride,
    output_feat_stride, eps, scale_by_weight: 'tl.constexpr', add_bias:
    'tl.constexpr', save_stats: 'tl.constexpr', BLOCK_SIZE_BATCH:
    'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Layer-normalizes the input.

    Args:
        input_pointer: Pointer to the input to layer-normalize.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to optional weights for affine transform.
            The weights, if provided, must be of shape [feat_dim].
        bias_pointer: Pointer to an optional bias vector for affine transform.
            The bias vector, if provided, must be of shape [feat_dim].
        mean_pointer: Pointer to an optional container the input's mean
            is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        inv_std_pointer: Pointer to an optional container the input's inverse
            standard deviation is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        scale_by_weight: Flag for scaling the normalized output by weights.
        add_bias: Flag for adding a bias vector to the normalized output
            if scale_by_weight is True.
        save_stats: Flag for saving the mean and standard deviation.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    input_pointer += input_batch_stride * batch_offset[:, None
        ] + input_feat_stride * feat_offset[None, :]
    output_pointer += output_batch_stride * batch_offset[:, None
        ] + output_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[
        None, :])
    mean = tl.sum(input, axis=1) / feat_dim
    diff = tl.where(feat_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / feat_dim + eps)
    if save_stats:
        tl.store(mean_pointer + batch_offset, mean, mask=batch_mask)
        tl.store(inv_std_pointer + batch_offset, inv_std, mask=batch_mask)
    output = diff * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        output *= weight
        if add_bias:
            bias = tl.load(bias_pointer + feat_offset, mask=feat_mask)
            output += bias
    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[
        None, :])

import triton
import triton.language as tl
import torch

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim',
    'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_forward_kernel(input_pointer, target_pointer, weight_pointer,
    sum_weights_pointer, output_pointer, batch_dim, spatial_dim,
    input_batch_stride, input_feat_stride, input_spatial_stride,
    target_batch_stride, target_spatial_stride, output_batch_stride,
    output_spatial_stride, reduction: 'tl.constexpr', weighted:
    'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_SPATIAL:
    'tl.constexpr'):
    """
    Measures the negative log likelihood loss between the input and target,
    with optional reweighing of each class.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to a container the sum of the class weights is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        output_pointer: Pointer to a container the loss is written to.
            The container must be of shape [batch_dim, spatial_dim] if reduction is 'none',
            and otherwise of shape [batch_dim/BLOCK_SIZE].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_spatial_stride: Stride necessary to jump one element along the
            output container's spatial dimension.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the summed weights and
            output container, which should later be summed.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)
    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim
    target_pointer += target_batch_stride * batch_offset[:, None
        ] + target_spatial_stride * spatial_offset[None, :]
    target = tl.load(target_pointer, mask=batch_mask[:, None] &
        spatial_mask[None, :])
    input_pointer += (input_feat_stride * target + input_batch_stride *
        batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :])
    input = tl.load(input_pointer, mask=batch_mask[:, None] & spatial_mask[
        None, :])
    output = -input
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask[:, None] &
            spatial_mask[None, :])
        output *= weight
    if reduction == 'none':
        output_pointer += output_batch_stride * batch_offset[:, None
            ] + output_spatial_stride * spatial_offset[None, :]
        tl.store(output_pointer, output, mask=batch_mask[:, None] &
            spatial_mask[None, :])
    elif reduction == 'mean':
        if weighted:
            tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
            tl.store(output_pointer + batch_pid, tl.sum(output))
        else:
            tl.store(output_pointer + batch_pid, tl.sum(output) / (
                batch_dim * spatial_dim))
    elif reduction == 'sum':
        tl.store(output_pointer + batch_pid, tl.sum(output))

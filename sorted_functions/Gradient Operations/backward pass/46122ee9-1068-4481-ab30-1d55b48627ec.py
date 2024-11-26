import triton
import triton.language as tl
import torch

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim',
    'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_backward_kernel(output_grad_pointer, target_pointer,
    weight_pointer, sum_weights_pointer, input_grad_pointer, batch_dim,
    spatial_dim, output_grad_batch_stride, output_grad_feat_stride,
    target_batch_stride, target_spatial_stride, input_grad_batch_stride,
    input_grad_feat_stride, input_grad_spatial_stride, reduction:
    'tl.constexpr', weighted: 'tl.constexpr', BLOCK_SIZE_BATCH:
    'tl.constexpr', BLOCK_SIZE_SPATIAL: 'tl.constexpr'):
    """
    Calculates the input gradient of negative log likelihood loss.

    Args:
        output_grad_pointer: Pointer to the loss's output gradients.
            The output gradients must be of shape [batch_dim, spatial_dim]
            if reduction is 'none', and otherwise [batch_dim/BLOCK_SIZE_BATCH].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to the sum of the class weights if the classes were weighed.
            The sum of weights must be a scalar.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim] and zeroed.
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
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
    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += output_grad_batch_stride * batch_offset[:, None
            ] + output_grad_feat_stride * spatial_offset[None, :]
        output_grad_mask = batch_mask[:, None] & spatial_mask[None, :]
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask)
    input_grad = -output_grad
    target_pointer += target_batch_stride * batch_offset[:, None
        ] + target_spatial_stride * spatial_offset[None, :]
    target = tl.load(target_pointer, mask=batch_mask[:, None] &
        spatial_mask[None, :])
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask[:, None] &
            spatial_mask[None, :])
        input_grad *= weight
        if reduction == 'mean':
            input_grad /= tl.load(sum_weights_pointer)
    elif reduction == 'mean':
        input_grad /= batch_dim * spatial_dim
    input_grad_pointer += (input_grad_feat_stride * target + 
        input_grad_batch_stride * batch_offset[:, None] + 
        input_grad_spatial_stride * spatial_offset[None, :])
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] &
        spatial_mask[None, :])

import triton
import triton.language as tl
import torch

@triton.jit
def triton_cross_entropy_forward_backward_kernel(logits_ptr, labels_ptr,
    grad_logits_ptr, losses_ptr, grad_losses, n_cols, logits_stride_0,
    grad_logits_stride_0, logits_scale_factor: 'tl.constexpr', block_size:
    'tl.constexpr'):
    block_idx = tl.program_id(0)
    col_offsets = tl.arange(0, block_size)
    logits_ptr = logits_ptr + block_idx * logits_stride_0
    mask = col_offsets < n_cols
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf'))
    if logits_scale_factor != 1.0:
        logits *= logits_scale_factor
    max_logits = tl.max(logits, 0)
    exp_logits = tl.exp(logits - max_logits)
    sum_exp_logits = tl.sum(exp_logits, 0)
    label_idx = tl.load(labels_ptr + block_idx)
    label_logits = tl.load(logits_ptr + label_idx)
    if label_idx < 0:
        loss = 0.0
    else:
        loss = tl.log(sum_exp_logits) + max_logits - label_logits
    tl.store(losses_ptr + block_idx, loss)
    grad_logits_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0
    col_offsets = tl.arange(0, block_size)
    label_idx = tl.load(labels_ptr + block_idx)
    exp_logits = exp_logits / sum_exp_logits
    if logits_scale_factor != 1.0:
        exp_logits *= logits_scale_factor
    if label_idx < 0:
        grad_losses = 0.0
    grad_logits = grad_losses * tl.where(col_offsets == label_idx, 
        exp_logits - 1.0, exp_logits)
    tl.store(grad_logits_ptr + col_offsets, grad_logits, mask=mask)

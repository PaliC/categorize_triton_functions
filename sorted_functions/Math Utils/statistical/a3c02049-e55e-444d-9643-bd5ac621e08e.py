import triton
import triton.language as tl
import torch

@triton.jit
def _kldiv_kernel_forward(y_ptr, y_stride, gt_ptr, gt_stride, loss_ptr,
    loss_stride, n_cols, eps, BLOCK_SIZE: 'tl.constexpr', log_target:
    'tl.constexpr'=False, reduction: 'tl.constexpr'=_REDUCTION_MODE_BATCHMEAN):
    pid = tl.program_id(0)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride
    base_offsets = tl.arange(0, BLOCK_SIZE)
    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)
    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)

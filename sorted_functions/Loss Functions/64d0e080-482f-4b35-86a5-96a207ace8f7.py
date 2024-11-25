import triton
import triton.language as tl
import torch

@triton.jit
def _kldiv_kernel_backward(target_ptr, target_stride, new_grads_ptr,
    new_grads_stride, n_cols, BLOCK_SIZE: 'tl.constexpr', log_target:
    'tl.constexpr'=False):
    pid = tl.program_id(0)
    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)
        tl.store(new_grads_ptr + offsets, res, mask=mask)

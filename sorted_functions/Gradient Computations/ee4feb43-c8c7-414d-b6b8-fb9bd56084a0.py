import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8)], key=['n_elements'],
    restore_value=['p_ptr', 'exp_avg_ptr'])
@triton.jit
def update_fn_kernel(p_ptr, grad_ptr, exp_avg_ptr, lr, wd, beta1, beta2,
    n_elements, BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
    p = p * (1 - lr * wd)
    diff = exp_avg - grad
    update = diff * beta1 + grad
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)
    p = p + update_sign * can_update
    exp_avg = diff * beta2 + grad
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)

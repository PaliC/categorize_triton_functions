import triton
import triton.language as tl
import torch

@triton.jit
def mars_adamw_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    prev_grad_ptr, lr, beta1, beta2, eps, weight_decay, gamma,
    max_grad_norm, step, bias_correction1, bias_correction2, n_elements,
    BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    prev_grad = tl.load(prev_grad_ptr + offsets, mask=mask)
    grad_diff = grad - prev_grad
    correction = gamma * beta1 / (1 - beta1) * grad_diff
    c_t = grad + correction
    c_t_norm = tl.sqrt(tl.sum(c_t * c_t))
    scale = tl.where(c_t_norm > max_grad_norm, max_grad_norm / c_t_norm, 1.0)
    c_t = c_t * scale
    exp_avg = beta1 * exp_avg + (1 - beta1) * c_t
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (c_t * c_t)
    tl.store(prev_grad_ptr + offsets, grad, mask=mask)
    step_size = lr / bias_correction1
    denom = tl.sqrt(exp_avg_sq) / tl.sqrt(bias_correction2) + eps
    update = exp_avg / denom
    param = param - step_size * (update + weight_decay * param)
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)

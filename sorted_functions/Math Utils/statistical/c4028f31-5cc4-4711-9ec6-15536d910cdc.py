import triton
import triton.language as tl
import torch

@triton.jit
def triton_adam_kernel(params_ptr, grads_ptr, exp_avgs_ptr, exp_avg_sqs_ptr,
    noop_flag_ptr, scale_ptr, step_size, beta1, beta2, bias_correction,
    decay_factor, epsilon, numel: 'tl.constexpr', block_size: 'tl.constexpr'):
    noop_flag = tl.load(noop_flag_ptr)
    if noop_flag != 0:
        return
    scale = tl.load(scale_ptr)
    block_start = tl.program_id(axis=0) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < numel
    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    grads = scale * grads
    exp_avgs = tl.load(exp_avgs_ptr + offsets, mask=mask)
    exp_avgs = beta1 * exp_avgs + (1 - beta1) * grads
    tl.store(exp_avgs_ptr + offsets, exp_avgs, mask=mask)
    exp_avg_sqs = tl.load(exp_avg_sqs_ptr + offsets, mask=mask)
    exp_avg_sqs = beta2 * exp_avg_sqs + (1 - beta2) * grads * grads
    tl.store(exp_avg_sqs_ptr + offsets, exp_avg_sqs, mask=mask)
    params = decay_factor * params - step_size * exp_avgs / (tl.sqrt(
        exp_avg_sqs) / bias_correction + epsilon)
    tl.store(params_ptr + offsets, params, mask=mask)

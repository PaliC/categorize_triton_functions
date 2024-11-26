import triton
import triton.language as tl
import torch

@triton.jit
def layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, y_ptr, N, eps:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x_offset = x_ptr + row_idx * N + cols
    x = tl.load(x_offset, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    y = x_centered * rstd * w + b
    tl.store(y_ptr + row_idx * N + cols, y, mask=mask)

import triton
import triton.language as tl
import torch

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = sqrt_2_over_pi * (1 + 0.044715 * x * x)
    y = 0.5 * x * (1 + x * coeff / (1 + tl.abs(x * coeff)))
    tl.store(y_ptr + offsets, y, mask=mask)

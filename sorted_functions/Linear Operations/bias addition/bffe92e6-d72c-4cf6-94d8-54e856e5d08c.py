import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: 'tl.constexpr'):
    off_x = tl.arange(0, B0)
    x = tl.load(x_ptr + off_x)
    x = x + 10.0
    tl.store(z_ptr + off_x, x)
    return

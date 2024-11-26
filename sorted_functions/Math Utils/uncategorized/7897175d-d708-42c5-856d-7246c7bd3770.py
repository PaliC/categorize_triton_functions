import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, size, block_size: 'tl.constexpr'):
    pid = tl.program_id(0)
    offsets = tl.arange(0, block_size) + pid * block_size
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask)

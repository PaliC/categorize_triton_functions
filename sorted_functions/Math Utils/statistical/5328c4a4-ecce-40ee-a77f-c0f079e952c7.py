import triton
import triton.language as tl
import torch

@triton.jit
def sum_kernel(y_ptr, x_ptr, size, block_size: 'tl.constexpr'):
    offsets = tl.arange(0, block_size)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask)
    y = tl.reduce(x, 0, combine_add)
    tl.store(y_ptr, y)

import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, length, output_ptr, BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

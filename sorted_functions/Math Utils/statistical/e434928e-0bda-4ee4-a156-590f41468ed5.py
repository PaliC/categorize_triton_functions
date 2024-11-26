import triton
import triton.language as tl
import torch

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

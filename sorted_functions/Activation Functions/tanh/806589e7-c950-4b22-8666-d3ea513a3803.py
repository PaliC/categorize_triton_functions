import triton
import triton.language as tl
import torch

@triton.jit
def tanh_kernel(x_ptr, length, output_ptr, BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    x = tl.load(x_ptr + offsets, mask=mask)
    output = libdevice.tanh(x)
    tl.store(output_ptr + offsets, output, mask=mask)

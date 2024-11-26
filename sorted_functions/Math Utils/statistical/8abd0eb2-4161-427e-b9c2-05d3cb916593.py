import triton
import triton.language as tl
import torch

@triton.jit
def add_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE:
    'tl.constexpr'):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + scalar
    tl.store(output_ptr + offsets, output, mask=mask)

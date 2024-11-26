import triton
import triton.language as tl
import torch

@triton.jit
def maximum_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE:
    'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, scalar)
    tl.store(output_ptr + offsets, output, mask=mask)
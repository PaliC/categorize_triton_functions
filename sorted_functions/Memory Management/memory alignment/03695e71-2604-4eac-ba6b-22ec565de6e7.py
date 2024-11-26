import triton
import triton.language as tl
import torch

@triton.jit
def atomic_kernel(x_ptr, increment):
    tl.atomic_add(x_ptr, increment)

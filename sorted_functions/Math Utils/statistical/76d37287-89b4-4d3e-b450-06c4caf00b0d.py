import triton
import triton.language as tl
import torch

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

import triton
import triton.language as tl
import torch

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

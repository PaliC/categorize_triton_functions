import triton
import triton.language as tl
import torch

@triton.jit
def tl_and_reduce_fn(a, b):
    return a & b

import triton
import triton.language as tl
import torch

@triton.jit
def _hash(x):
    x = (x >> 16 ^ x) * 73244475
    x = (x >> 16 ^ x) * 73244475
    x = x >> 16 ^ x
    return x

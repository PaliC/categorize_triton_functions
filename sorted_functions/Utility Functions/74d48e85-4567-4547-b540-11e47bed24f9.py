import triton
import triton.language as tl
import torch

@triton.jit
def pair_hash(x, h):
    h = h ^ x
    h = (h << 24) + h * 403
    return h

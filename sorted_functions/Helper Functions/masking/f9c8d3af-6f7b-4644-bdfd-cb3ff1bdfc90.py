import triton
import triton.language as tl
import torch

@triton.jit
def mask_1d(offs, max):
    return offs < max

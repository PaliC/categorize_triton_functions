import triton
import triton.language as tl
import torch

@triton.jit
def _floor(x):
    return x - x % 1

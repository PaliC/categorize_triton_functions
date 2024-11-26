import triton
import triton.language as tl
import torch

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

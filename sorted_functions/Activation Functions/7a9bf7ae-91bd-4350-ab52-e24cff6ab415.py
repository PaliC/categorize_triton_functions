import triton
import triton.language as tl
import torch

@triton.jit
def _softplus(x):
    z = tl.where(x >= 0, x + tl.log(1 + tl.exp(-x)), tl.log(1 + tl.exp(x)))
    return z

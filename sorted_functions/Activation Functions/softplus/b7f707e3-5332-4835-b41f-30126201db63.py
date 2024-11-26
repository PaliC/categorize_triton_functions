import triton
import triton.language as tl
import torch

@triton.jit
def _d_softplus(grad, x):
    z = tl.where(x >= 0, 1 / (1 + tl.exp(-x)), 1 - 1 / (1 + tl.exp(x)))
    return grad * z

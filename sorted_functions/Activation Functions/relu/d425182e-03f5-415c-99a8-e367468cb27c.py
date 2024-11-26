import triton
import triton.language as tl
import torch

@triton.jit
def relu_grad(x):
    zero = 0.0
    one = 1.0
    return tl.where(x >= 0, one, zero)

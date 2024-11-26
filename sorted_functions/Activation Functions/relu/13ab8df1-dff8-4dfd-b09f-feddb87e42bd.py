import triton
import triton.language as tl
import torch

@triton.jit
def relu_grad(x):
    return tl.where(x >= 0, 1.0, 0.0)

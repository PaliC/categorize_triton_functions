import triton
import triton.language as tl
import torch

@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0)

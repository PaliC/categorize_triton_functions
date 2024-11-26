import triton
import triton.language as tl
import torch

@triton.jit
def relu(x):
    return tl.max(x, 0.0)

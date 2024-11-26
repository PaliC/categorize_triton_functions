import triton
import triton.language as tl
import torch

@triton.jit
def identity(x):
    return x

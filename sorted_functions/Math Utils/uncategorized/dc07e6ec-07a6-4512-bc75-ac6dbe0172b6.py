import triton
import triton.language as tl
import torch

@triton.jit
def combine_add(a, b):
    return a + b

import triton
import triton.language as tl
import torch

@triton.jit
def silu(input):
    return input * tl.sigmoid(input)

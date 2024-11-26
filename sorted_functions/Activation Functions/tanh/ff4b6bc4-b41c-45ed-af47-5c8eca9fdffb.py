import triton
import triton.language as tl
import torch

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

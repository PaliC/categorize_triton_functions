import triton
import triton.language as tl
import torch

@triton.jit
def add_grad(left, right):
    right = triton_unbroadcast(right, left.shape)
    return left + right

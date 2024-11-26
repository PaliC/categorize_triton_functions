import triton
import triton.language as tl
import torch

@triton.jit
def leaky_relu_grad(x):
    min_grad = 0.01
    max_grad = 1
    min_grad = min_grad
    max_grad = max_grad
    return tl.where(x >= 0, max_grad, min_grad)

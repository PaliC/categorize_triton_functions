import triton
import triton.language as tl
import torch

@triton.jit
def d_sigmoid(dy, x):
    s = tl.sigmoid(x)
    return dy * s * (1 - s)

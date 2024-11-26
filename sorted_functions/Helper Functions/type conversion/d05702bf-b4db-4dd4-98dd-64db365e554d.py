import triton
import triton.language as tl
import torch

@triton.jit
def zeroslike(x):
    return tl.zeros(x.shape, tl.float32)

import triton
import triton.language as tl
import torch

@triton.jit
def _round(x):
    return _floor(x + 0.5)

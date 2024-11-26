import triton
import triton.language as tl
import torch

@triton.jit
def logaddexp(a, b):
    tmp = a - b
    return tl.where(tmp > 0, tl.log(tl.exp(b - a) + 1) + a, tl.log(tl.exp(a -
        b) + 1) + b)

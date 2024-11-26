import triton
import triton.language as tl
import torch

@triton.jit
def demo1(x_ptr):
    range = tl.arange(0, 8)
    None
    x = tl.load(x_ptr + range, range < 5, 0)
    None

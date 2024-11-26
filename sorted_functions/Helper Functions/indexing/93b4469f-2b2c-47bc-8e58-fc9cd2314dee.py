import triton
import triton.language as tl
import torch

@triton.jit
def demo2(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range * 4 + j_range
    None
    x = tl.load(x_ptr + range, (i_range < 4) & (j_range < 3), 0)
    None

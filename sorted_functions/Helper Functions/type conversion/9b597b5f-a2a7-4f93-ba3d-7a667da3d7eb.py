import triton
import triton.language as tl
import torch

@triton.jit
def update_position(a, b, position_a, position_b):
    tmp = a - b
    return tl.where(tmp > 0, position_a, position_b)

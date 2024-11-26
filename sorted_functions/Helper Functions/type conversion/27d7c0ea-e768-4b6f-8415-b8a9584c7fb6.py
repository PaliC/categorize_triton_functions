import triton
import triton.language as tl
import torch

@triton.jit
def int32_to_float01(x):
    x_01 = (x + MAX_INT_32_F + MAX_UINT_32_F_EPS) / (MAX_UINT_32_F +
        MAX_UINT_32_F_EPS)
    return x_01

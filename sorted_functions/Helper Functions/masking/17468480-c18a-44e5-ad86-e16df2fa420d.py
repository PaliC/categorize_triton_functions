import triton
import triton.language as tl
import torch

@triton.jit
def mask_2d(offs0, offs1, max0, max1):
    return (tl.expand_dims(offs0, 1) < max0) & (tl.expand_dims(offs1, 0) < max1
        )

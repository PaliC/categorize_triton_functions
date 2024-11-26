import triton
import triton.language as tl
import torch

@triton.jit
def offset_2d(offs0, offs1, stride0, stride1=1):
    return tl.expand_dims(offs0, 1) * stride0 + tl.expand_dims(offs1, 0
        ) * stride1

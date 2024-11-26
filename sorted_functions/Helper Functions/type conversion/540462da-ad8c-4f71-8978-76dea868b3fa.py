import triton
import triton.language as tl
import torch

@triton.jit
def pack64(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.uint32, bitcast=True)
    a = a << 32
    b = b.to(dtype=tl.uint32, bitcast=True)
    return a | b

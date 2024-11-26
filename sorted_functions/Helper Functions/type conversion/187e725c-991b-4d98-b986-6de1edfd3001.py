import triton
import triton.language as tl
import torch

@triton.jit
def unpack64(merged):
    tl.static_assert(merged.dtype == tl.uint64)
    b = (merged & 4294967295).to(tl.uint32)
    a = (merged >> 32).to(tl.uint32)
    return a, b

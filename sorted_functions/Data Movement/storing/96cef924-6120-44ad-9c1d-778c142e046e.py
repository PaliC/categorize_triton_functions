import triton
import triton.language as tl
import torch

@triton.jit
def store_full_1d(vals, ptr, sz: 'const', stride=1):
    """Store 1d block into vector (defined by ptr)"""
    offs = offset_1d(sz)
    mask = mask_1d(offs, sz)
    tl.store(ptr + offs, vals, mask)

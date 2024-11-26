import triton
import triton.language as tl
import torch

@triton.jit
def store_full_2d(vals, ptr, sz0: 'const', sz1: 'const', stride0=None,
    stride1=1):
    """Store 2d block into matrix (defined by ptr)"""
    stride0 = stride0 or sz1
    offs = offset_2d(tl.arange(0, sz0), tl.arange(0, sz1), stride0, stride1)
    mask = mask_2d(tl.arange(0, sz0), tl.arange(0, sz1), sz0, sz1)
    tl.store(ptr + offs, vals, mask)

import triton
import triton.language as tl
import torch

@triton.jit
def store_2d(vals, ptr, sz0: 'const', sz1: 'const', n0, n1, max0, max1,
    stride0=None, stride1=1):
    """Store 2d block into (n0,n1)th chunk of matrix (defined by ptr), where each chunk has size (sz0, sz1)"""
    stride0 = stride0 or sz1
    offs0 = offset_1d(sz0, n0)
    offs1 = offset_1d(sz1, n1)
    offs = offset_2d(offs0, offs1, stride0, stride1)
    mask = mask_2d(offs0, offs1, max0, max1)
    tl.store(ptr + offs, vals, mask)

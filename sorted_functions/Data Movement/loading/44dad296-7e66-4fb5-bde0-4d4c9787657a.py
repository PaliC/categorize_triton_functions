import triton
import triton.language as tl
import torch

@triton.jit
def load_full_1d(ptr, sz: 'const', stride=1):
    """Load 1d block [0,...,sz-1]"""
    offs = offset_1d(sz)
    mask = mask_1d(offs, sz)
    return tl.load(ptr + offs, mask)

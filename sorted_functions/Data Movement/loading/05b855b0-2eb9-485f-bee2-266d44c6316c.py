import triton
import triton.language as tl
import torch

@triton.jit
def load_full_2d(ptr, sz0: 'const', sz1: 'const', stride0=None, stride1=1):
    """Load 2d block [0,...,sz0-1] x [0,...,sz1-1] """
    stride0 = stride0 or sz1
    offs = offset_2d(tl.arange(0, sz0), tl.arange(0, sz1), stride0, stride1)
    mask = mask_2d(tl.arange(0, sz0), tl.arange(0, sz1), sz0, sz1)
    return tl.load(ptr + offs, mask)

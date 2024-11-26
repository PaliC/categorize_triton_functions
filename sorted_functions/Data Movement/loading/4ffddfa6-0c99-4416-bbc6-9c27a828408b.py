import triton
import triton.language as tl
import torch

@triton.jit
def load_1d(ptr, sz: 'const', n, max, stride=1):
    """Chunk 1d vector (defined by ptr) into 1d grid, where each chunk has size sz. Load the nth chunk. Ie, load [n*sz,...,(n+1)*sz-1]."""
    offs = offset_1d(sz, n)
    mask = mask_1d(offs, max)
    return tl.load(ptr + offs, mask)

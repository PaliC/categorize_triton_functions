import triton
import triton.language as tl
import torch

@triton.jit
def store_1d(vals, ptr, sz: 'const', n, max, stride=1):
    """Store 1d block into nth chunk of vector (defined by ptr), where each chunk has size sz"""
    offs = offset_1d(sz, n)
    mask = mask_1d(offs, max)
    tl.store(ptr + offs, vals, mask)

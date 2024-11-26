import triton
import triton.language as tl
import torch

@triton.jit
def offset_1d(sz: 'const', n_prev_chunks=0):
    return n_prev_chunks * sz + tl.arange(0, sz)

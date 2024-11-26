import triton
import triton.language as tl
import torch

@triton.jit
def dropout_offsets(philox_seed, philox_offset, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]

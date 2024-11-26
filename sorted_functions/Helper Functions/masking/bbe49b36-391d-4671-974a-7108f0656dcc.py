import triton
import triton.language as tl
import torch

@triton.jit
def dropout_rng(philox_seed, philox_offset, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, m, n, stride)
    return tl.rand(philox_seed, rng_offsets)

import triton
import triton.language as tl
import torch

@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep

import triton
import triton.language as tl
import torch

@triton.jit
def _int_to_randn(x1, x2, seed):
    x_hash_1 = _hash(x1)
    x_hash_2 = _hash(x2)
    x_hash_1 = _pair_hash(_pair_hash(2166136261, seed), x_hash_1)
    x_hash_2 = _pair_hash(_pair_hash(2166136261, seed + 1), x_hash_2)
    x_01_1 = (x_hash_1 + 10) / (4294967295.0 + 10)
    x_01_2 = (x_hash_2 + 10) / (4294967295.0 + 10)
    z = tl.sqrt(-2 * tl.log(x_01_1)) * tl.cos(6.28318530718 * x_01_2)
    return z

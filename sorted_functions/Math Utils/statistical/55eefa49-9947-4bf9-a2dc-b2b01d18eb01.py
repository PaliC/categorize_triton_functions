import triton
import triton.language as tl
import torch

@triton.jit
def int_to_randn(x1, x2, seed):
    x_hash_1 = hash(x1)
    x_hash_2 = hash(x2)
    x_hash_1 = pair_hash(pair_hash(INT32_PRIME, seed), x_hash_1)
    x_hash_2 = pair_hash(pair_hash(INT32_PRIME, seed + 1), x_hash_2)
    x_01_1 = int32_to_float01(x_hash_1)
    x_01_2 = int32_to_float01(x_hash_2)
    z = tl.sqrt(-2 * tl.log(x_01_1)) * tl.cos(6.28318530718 * x_01_2)
    return z

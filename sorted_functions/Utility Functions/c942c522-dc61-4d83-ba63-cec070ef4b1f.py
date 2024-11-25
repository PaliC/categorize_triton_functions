import triton
import triton.language as tl
import torch

@triton.jit
def _contract_pi_one(x, n):
    x_c = tl.where(n <= 1.0, x, tl.where(tl.abs(tl.abs(x) - n) <= 1e-08, (2 -
        1 / tl.abs(x)) * (x / tl.abs(x)), x / n))
    x_c = x_c * 0.5
    return x_c

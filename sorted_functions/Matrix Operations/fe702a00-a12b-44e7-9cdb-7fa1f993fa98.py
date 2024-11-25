import triton
import triton.language as tl
import torch

@triton.jit
def contract_pi(x, y, z):
    n = tl.maximum(tl.maximum(tl.abs(x), tl.abs(y)), tl.abs(z))
    x_c = _contract_pi_one(x, n)
    y_c = _contract_pi_one(y, n)
    z_c = _contract_pi_one(z, n)
    return x_c, y_c, z_c

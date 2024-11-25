import triton
import triton.language as tl
import torch

@triton.jit
def _contract_pi(x, y, z, perc_foreground):
    n = tl.maximum(tl.maximum(tl.abs(x), tl.abs(y)), tl.abs(z))
    x_c = _contract_pi_one(x, n, perc_foreground)
    y_c = _contract_pi_one(y, n, perc_foreground)
    z_c = _contract_pi_one(z, n, perc_foreground)
    return x_c, y_c, z_c

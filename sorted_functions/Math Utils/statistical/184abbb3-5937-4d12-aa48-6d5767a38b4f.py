import triton
import triton.language as tl
import torch

@triton.jit
def _contract_pi_one(x, n, perc_foreground):
    x_c = tl.where(n <= 1.0, x, tl.where(tl.abs(tl.abs(x) - n) <= 1e-08, (
        1.0 / perc_foreground - (1.0 / perc_foreground - 1) / tl.abs(x)) *
        (x / tl.abs(x)), x / n))
    x_c = x_c * perc_foreground
    return x_c

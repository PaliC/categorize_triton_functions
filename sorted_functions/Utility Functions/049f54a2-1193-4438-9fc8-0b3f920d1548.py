import triton
import triton.language as tl
import torch

@triton.jit
def _depth_lin(near, far, n, step):
    frac_step = step / (n - 1)
    return (far - near) * frac_step + near

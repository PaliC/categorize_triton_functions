import triton
import triton.language as tl
import torch

@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))

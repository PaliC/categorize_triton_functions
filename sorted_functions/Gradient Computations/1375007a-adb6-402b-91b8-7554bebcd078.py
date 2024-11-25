import triton
import triton.language as tl
import torch

@triton.jit
def gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))
    pdf = tl.exp(-0.5 * x * x) * _gaussian_pdf_normalization
    return cdf + x * pdf

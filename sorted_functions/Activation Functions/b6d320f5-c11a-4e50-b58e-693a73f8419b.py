import triton
import triton.language as tl
import torch

@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x *
        x * x)))

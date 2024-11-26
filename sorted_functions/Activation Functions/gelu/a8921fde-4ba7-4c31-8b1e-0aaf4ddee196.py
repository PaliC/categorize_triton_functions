import triton
import triton.language as tl
import torch

@triton.jit
def gelu(x):
    c = 0.7978845608028654
    x_cubed = x * x * x
    tanh_arg = c * (x + 0.044715 * x_cubed)
    tanh_result = tanh(tanh_arg)
    return 0.5 * x * (1 + tanh_result)

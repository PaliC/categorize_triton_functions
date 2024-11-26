import triton
import triton.language as tl
import torch

@triton.jit
def gelu_approx_grad(x):
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 
        0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)

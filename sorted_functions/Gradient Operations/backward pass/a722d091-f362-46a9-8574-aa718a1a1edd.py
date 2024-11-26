import triton
import triton.language as tl
import torch

@triton.jit
def d_linear_relu(d_y, w, b, xwb, x):
    d_y_relu = d_y * (xwb > 0.0)
    return d_linear(d_y_relu, w, b, x)

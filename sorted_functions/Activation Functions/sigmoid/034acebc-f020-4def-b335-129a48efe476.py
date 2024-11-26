import triton
import triton.language as tl
import torch

@triton.jit
def _d_color_activation(dy, x):
    return dy * tl.sigmoid(x) * (1 - tl.sigmoid(x))

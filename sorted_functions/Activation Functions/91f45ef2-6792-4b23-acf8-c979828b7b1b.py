import triton
import triton.language as tl
import torch

@triton.jit
def _color_activation(x):
    return tl.sigmoid(x)

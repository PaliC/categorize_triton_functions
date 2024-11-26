import triton
import triton.language as tl
import torch

@triton.jit
def relu(x):
    """ReLU(Rectified Linear Unit, 修正线性单元), only support inference.
    max(0, x)
    """
    return tl.maximum(0, x)

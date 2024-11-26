import triton
import triton.language as tl
import torch

@triton.jit
def leaky_relu(x):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    scale = 0.01
    scale = scale
    return tl.where(x >= 0, x, scale * x)

import triton
import triton.language as tl
import torch

@triton.jit
def tl_tanh(a: 'tl.tensor') ->tl.tensor:
    return tl_libdevice.tanh(a)

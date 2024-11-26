import triton
import triton.language as tl
import torch

@triton.jit
def tl_log1p(a: 'tl.tensor') ->tl.tensor:
    return tl_libdevice.log1p(a)

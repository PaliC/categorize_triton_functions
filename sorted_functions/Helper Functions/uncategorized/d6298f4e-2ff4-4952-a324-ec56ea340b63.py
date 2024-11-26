import triton
import triton.language as tl
import torch

@triton.jit
def nop_kernel():
    pass

import triton
import triton.language as tl
import torch

@triton.jit
def breakpoint_once():
    breakpoint_if('=0,=0,=0')

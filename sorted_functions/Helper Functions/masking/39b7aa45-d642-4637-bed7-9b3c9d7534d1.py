import triton
import triton.language as tl
import torch

@triton.jit
def print_once(*txt):
    print_if(*txt, conds='=0,=0,=0')

import triton
import triton.language as tl
import torch

@triton.jit
def print_line(str_line):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        None

import triton
import triton.language as tl
import torch

@triton.jit
def print_grid():
    pid = tl.program_id(0)
    tl.device_print('pid: ', pid)

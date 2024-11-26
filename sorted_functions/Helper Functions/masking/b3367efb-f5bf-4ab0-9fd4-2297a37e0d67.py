import triton
import triton.language as tl
import torch

@triton.jit
def print_if(*txt, conds):
    """Print txt, if condition on pids is fulfilled"""
    if test_pid_conds(conds):
        None

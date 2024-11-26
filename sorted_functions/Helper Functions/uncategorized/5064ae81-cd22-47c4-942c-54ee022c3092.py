import triton
import triton.language as tl
import torch

@triton.jit
def breakpoint_if(conds):
    """Stop kernel, if condition on pids is fulfilled"""
    if test_pid_conds(conds):
        set_trace()

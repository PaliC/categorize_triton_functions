import triton
import triton.language as tl
import torch

@triton.jit
def offsets_from_base(ptrs, base_ptr):
    """Return offsets for which ptrs = base_ptr + offsets"""
    return ptrs - base_ptr

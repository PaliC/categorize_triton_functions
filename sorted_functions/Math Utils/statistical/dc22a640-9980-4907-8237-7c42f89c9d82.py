import triton
import triton.language as tl
import torch

@triton.jit
def _abs_max(val1, val2):
    val1_abs = tl.abs(val1)
    val2_abs = tl.abs(val2)
    if val1_abs >= val2_abs:
        return val1_abs
    else:
        return val2_abs

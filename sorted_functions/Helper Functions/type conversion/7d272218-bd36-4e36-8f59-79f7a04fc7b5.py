import triton
import triton.language as tl
import torch

@triton.jit
def cast_uint32_to_half2(scale_shift):
    """Extract two float16 packed into one int32"""
    scale = scale_shift & 65535
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16)
    shift = shift.to(tl.uint16)
    return scale, shift

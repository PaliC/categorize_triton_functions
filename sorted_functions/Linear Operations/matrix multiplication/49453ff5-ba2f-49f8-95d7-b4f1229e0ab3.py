import triton
import triton.language as tl
import torch

@triton.jit
def _dot_tf32_f32_3x(a, b, trans_a=False, trans_b=False):
    """Perform the 3-pass tf32 dot function."""
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a_ = a.to(tl.uint32, bitcast=True) & 4294959104
    b_ = b.to(tl.uint32, bitcast=True) & 4294959104
    a_err = a - a_
    b_err = b - b_
    if trans_a:
        a_ = tl.trans(a_)
        a_err = tl.trans(a_err)
    if trans_b:
        b_ = tl.trans(b_)
        b_err = tl.trans(b_err)
    return tl.dot(a_, b_, out_dtype=tl.float32) + (tl.dot(a_, b_err,
        out_dtype=tl.float32) + tl.dot(a_err, b_, out_dtype=tl.float32))

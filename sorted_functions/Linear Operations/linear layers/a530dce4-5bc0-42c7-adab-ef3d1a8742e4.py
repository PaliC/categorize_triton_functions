import triton
import triton.language as tl
import torch

@triton.jit
def d_linear(d_y, w, b, x):
    d_x = tl.dot(d_y, tl.trans(w), allow_tf32=ALLOW_TF32)
    d_w = tl.dot(tl.trans(d_y), x, allow_tf32=ALLOW_TF32)
    d_b = tl.sum(d_y, axis=0)
    return d_x, d_w, d_b

import triton
import triton.language as tl
import torch

@triton.jit
def ub1(X, Y):
    r = tl.arange(0, 16)
    r2 = tl.arange(0, 32)
    x = tl.load(X + 16 * r2[:, None] + r)
    y = triton_unbroadcast(x, tl.arange(0, 16).shape)
    tl.store(Y + r, y)

import triton
import triton.language as tl
import torch

@triton.jit
def tr1(X, Y):
    r = tl.arange(0, 16)
    x = tl.load(X + r)
    y = comp2tt(x)
    tl.store(Y + 16 * r[:, None] + r, y)

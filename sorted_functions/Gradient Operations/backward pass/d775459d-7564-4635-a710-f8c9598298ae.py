import triton
import triton.language as tl
import torch

@triton.jit
def tr2(X, dX, dY):
    r = tl.arange(0, 16)
    r2 = tl.arange(0, 16)[:, None]
    x = tl.load(X + r)
    dy = tl.load(dY + 16 * r2 + r)
    tl.static_print('shape', dy.shape)
    dx = dcomp2dx(x, dy)
    tl.static_print('shape', dx.shape)
    tl.store(dX + r, dx)

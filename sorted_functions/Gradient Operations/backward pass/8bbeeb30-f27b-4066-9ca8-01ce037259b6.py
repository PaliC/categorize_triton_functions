import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_bwd_dx(DX, DY, X, Mean, Rstd, stride_dx, stride_dy,
    stride_x, D, eps, BLOCK_D: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row * stride_x
    DY += row * stride_dy
    DX += row * stride_dx
    x = tl.load(X + cols, mask=mask, other=0)
    dy = tl.load(DY + cols, mask=mask, other=0)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)
    c1 = tl.sum(xhat * dy, axis=0) / D
    c2 = tl.sum(dy, axis=0) / D
    dx = (dy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)

import triton
import triton.language as tl
import torch

@triton.jit
def _weighted_rms_norm_bwd_dx(DX, DY, DW, X, W, Rstd, Lock, stride_dx,
    stride_dy, stride_x, D, eps, GROUP_N, BLOCK_D: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row * stride_x
    DY += row * stride_dy
    DX += row * stride_dx
    x = tl.load(X + cols, mask=mask, other=0)
    dy = tl.load(DY + cols, mask=mask, other=0)
    rstd = tl.load(Rstd + row)
    xhat = x * rstd
    w = tl.load(W + cols, mask=mask)
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / D
    dx = (wdy - xhat * c1) * rstd
    tl.store(DX + cols, dx, mask=mask)
    lock_id = row % GROUP_N
    Lock += lock_id
    Count = Lock + GROUP_N
    DW = DW + lock_id * D + cols
    partial_dw = dy * xhat
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.atomic_xchg(Lock, 0)

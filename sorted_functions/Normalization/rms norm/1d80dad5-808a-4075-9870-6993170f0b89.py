import triton
import triton.language as tl
import torch

@triton.jit
def _rms_layer_norm_bwd_dx_fused(DX, DY, DW, X, W, RMS, Lock, stride, N,
    GROUP_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0)
    dy = tl.load(DY + cols, mask=mask, other=0)
    w = tl.load(W + cols, mask=mask)
    rms = tl.load(RMS + row)
    xhat = x * rms
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * c1) * rms
    tl.store(DX + cols, dx, mask=mask)
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

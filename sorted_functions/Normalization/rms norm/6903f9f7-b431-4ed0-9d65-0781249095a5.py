import triton
import triton.language as tl
import torch

@triton.jit
def srms_norm_bwd_dx_fused(DX, DY, X, V, stride, N, BLOCK_SIZE_N:
    'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    rstd = tl.load(V + row)
    xhat = x * rstd
    wdy = dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean1) * rstd
    mask = cols < N
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

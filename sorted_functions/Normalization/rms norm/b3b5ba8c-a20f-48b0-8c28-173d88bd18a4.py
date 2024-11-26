import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_bwd(dY, dX, dW, X, W, Rstd, stride, N, BLOCK_SIZE: 'tl.constexpr'
    ):
    row = tl.program_id(0)
    X += row * stride
    dY += row * stride
    dX += row * stride
    dW += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(dY + cols, mask=mask, other=0.0)
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    rstd = tl.load(Rstd + row)
    m = dy * w
    dx = rstd * m
    dx += rstd * -(1 / N) * rstd * rstd * tl.sum(m * x, axis=0) * x
    dw = dy * (x * rstd)
    tl.store(dX + cols, dx, mask=mask)
    tl.store(dW + cols, dw, mask=mask)

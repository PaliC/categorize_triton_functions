import triton
import triton.language as tl
import torch

@triton.jit
def _rmsnorm_fwd_kernel(X, Y, W, Rstd, stride_x_row, stride_y_row, N, eps,
    BLOCK_N: 'tl.constexpr', IS_EVEN_N: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    if IS_EVEN_N:
        w = tl.load(W + cols)
    else:
        w = tl.load(W + cols, mask=mask)
    x_hat = x * rstd
    y = x_hat * w
    if IS_EVEN_N:
        tl.store(Y + cols, y)
    else:
        tl.store(Y + cols, y, mask=mask)

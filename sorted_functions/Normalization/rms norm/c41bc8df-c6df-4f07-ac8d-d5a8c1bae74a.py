import triton
import triton.language as tl
import torch

@triton.jit
def _weighted_rms_norm_fwd(X, Y, W, Rstd, D, eps, stride_x, stride_y,
    BLOCK_D: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(X + cols, mask=cols < D, other=0.0)
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask)
    y = y * w
    tl.store(Y + cols, y, mask=mask)

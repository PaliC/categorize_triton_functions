import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N'])
@triton.jit
def _rms_norm_fwd_kernel(X, stride_x, Y, stride_y, W, Rstd, eps, M, N,
    block_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, block_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)

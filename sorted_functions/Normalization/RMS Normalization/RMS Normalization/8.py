import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_fwd_fused(X, Y, W, x_stride0, x_stride1, y_stride0, y_stride1,
    N, eps, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    Y += row * y_stride0
    X += row * x_stride0
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols * x_stride1, mask=cols < N, other=0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0)
        x_hat = x * rstd
        y = x_hat * w
        tl.store(Y + cols * y_stride1, y, mask=mask)

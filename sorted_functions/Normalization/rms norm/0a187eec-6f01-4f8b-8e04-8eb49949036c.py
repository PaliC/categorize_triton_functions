import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_dquant_kernel(X, Y, W, scale, stride, N, eps, BLOCK_SIZE:
    'tl.constexpr'):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    _max_x = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0)
        w = tl.load(W + cols, mask=mask)
        norm = x * rstd * w
        _max_x = tl.maximum(_max_x, tl.max(tl.abs(norm), axis=0))
    scale_x = _max_x / 127.0
    tl.store(scale + row, scale_x)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0)
        w = tl.load(W + cols, mask=mask)
        norm = x * rstd * w
        norm = norm / scale_x
        norm = tl.where(norm > 0, norm + 0.5, norm - 0.5)
        tl.store(Y + cols, norm, mask=mask)

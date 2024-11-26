import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_dquant_kernel(X, Y, W, B, out, scale, stride, N, eps,
    BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    out += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    _max_x = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0)
        _norm = (x - mean) * rstd * w + b
        tl.store(out + cols, _norm, mask=mask)
        _max_x = tl.maximum(_max_x, tl.max(tl.abs(_norm), axis=0))
    scale_x = _max_x / 127.0
    tl.store(scale + row, scale_x)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        _norm = tl.load(out + cols, mask=mask, other=0.0)
        _norm = _norm / scale_x + 0.5
        tl.store(Y + cols, _norm, mask=mask)

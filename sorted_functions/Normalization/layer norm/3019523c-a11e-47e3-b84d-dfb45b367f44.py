import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_modulation_fwd(X, Y, W, B, Mean, Rstd, stride, seq_len, N,
    eps, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    batch_idx = row // seq_len
    Y += row * stride
    X += row * stride
    W += batch_idx * stride
    B += batch_idx * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    var = tl.sum(x * x, axis=0) / N - mean * mean
    rstd = tl.rsqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    y = (x - mean) * rstd * (1 + w) + b
    tl.store(Y + cols, y, mask=mask)

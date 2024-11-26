import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd(X, Y, Mean, Rstd, D, eps, stride_x, stride_y, TRAINING:
    'tl.constexpr', BLOCK_D: 'tl.constexpr', COMPUTE_MEAN_AND_RSTD:
    'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(X + cols, mask=cols < D, other=0.0)
    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x, axis=0) / D
    else:
        mean = tl.load(Mean + row)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    if COMPUTE_MEAN_AND_RSTD:
        _var = tl.zeros([BLOCK_D], dtype=tl.float32)
        _var += x_mean * x_mean
        var = tl.sum(_var, axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Mean + row, mean)
            tl.store(Rstd + row, rstd)
    else:
        rstd = tl.load(Rstd + row)
    mask = cols < D
    y = x_mean * rstd
    tl.store(Y + cols, y, mask=mask)

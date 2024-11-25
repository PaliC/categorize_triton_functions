import triton
import triton.language as tl
import torch

@triton.heuristics({'HAS_BIAS': lambda args: args['B'] is not None})
@triton.heuristics({'HAS_Z': lambda args: args['Z'] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(X, Y, W, B, Z, Mean, Rstd, stride_x_row,
    stride_y_row, stride_z_row, M, N, eps, BLOCK_N: 'tl.constexpr',
    HAS_BIAS: 'tl.constexpr', HAS_Z: 'tl.constexpr', NORM_BEFORE_GATE:
    'tl.constexpr', IS_RMS_NORM: 'tl.constexpr'):
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask)
        y *= z * tl.sigmoid(z)
    tl.store(Y + cols, y, mask=mask)

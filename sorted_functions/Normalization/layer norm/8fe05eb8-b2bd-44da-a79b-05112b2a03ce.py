import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT',
    'IS_RMS_NORM', 'HAS_BIAS'])
@triton.jit
def _layer_norm_fwd_quant_kernel(X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean,
    Rstd, stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, N,
    eps, IS_RMS_NORM: 'tl.constexpr', BLOCK_N: 'tl.constexpr', HAS_RESIDUAL:
    'tl.constexpr', STORE_RESIDUAL_OUT: 'tl.constexpr', HAS_WEIGHT:
    'tl.constexpr', HAS_BIAS: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
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
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b
    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-05)
    y = tl.math.round(y * scale)
    y = tl.maximum(tl.minimum(y, 127), -128) / scale
    tl.store(Y + cols, y, mask=mask)

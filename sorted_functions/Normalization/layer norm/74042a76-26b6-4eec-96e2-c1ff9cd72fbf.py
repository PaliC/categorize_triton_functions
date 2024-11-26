import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT',
    'IS_RMS_NORM', 'HAS_BIAS'])
@triton.heuristics({'HAS_X1': lambda args: args['X1'] is not None})
@triton.heuristics({'HAS_W1': lambda args: args['W1'] is not None})
@triton.heuristics({'HAS_B1': lambda args: args['B1'] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(X, Y, W, B, RESIDUAL, X1, W1, B1, Y1,
    RESIDUAL_OUT, ROWSCALE, SEEDS, DROPOUT_MASK, Mean, Rstd, stride_x_row,
    stride_y_row, stride_res_row, stride_res_out_row, stride_x1_row,
    stride_y1_row, M, N, eps, dropout_p, IS_RMS_NORM: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', HAS_RESIDUAL: 'tl.constexpr',
    STORE_RESIDUAL_OUT: 'tl.constexpr', HAS_BIAS: 'tl.constexpr',
    HAS_DROPOUT: 'tl.constexpr', STORE_DROPOUT_MASK: 'tl.constexpr',
    HAS_ROWSCALE: 'tl.constexpr', HAS_X1: 'tl.constexpr', HAS_W1:
    'tl.constexpr', HAS_B1: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row)
        x *= rowscale
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row)
            x1 *= rowscale
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + M + row), cols, n_rounds=7
                ) > dropout_p
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask,
                    mask=cols < N)
        x += x1
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
    w = tl.load(W + cols, mask=mask)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)

import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL',
    'IS_RMS_NORM', 'HAS_BIAS'])
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['Y'] is not None})
@triton.jit
def _layer_norm_bwd_kernel(X, W, B, Y, DY, DX, DW, DB, DRESIDUAL,
    DRESIDUAL_IN, Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row,
    stride_dx_row, stride_dres_row, stride_dres_in_row, M, N, G,
    rows_per_program, programs_per_group, IS_RMS_NORM: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', HAS_DRESIDUAL: 'tl.constexpr', STORE_DRESIDUAL:
    'tl.constexpr', HAS_WEIGHT: 'tl.constexpr', HAS_BIAS: 'tl.constexpr',
    RECOMPUTE_OUTPUT: 'tl.constexpr'):
    row_block_id = tl.program_id(0)
    group_id, program_id_in_group = (row_block_id // programs_per_group, 
        row_block_id % programs_per_group)
    row_start = group_id + program_id_in_group * G * rows_per_program
    row_end = min(row_start + G * rows_per_program, M)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + group_id * stride_x_row + cols, mask=mask)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + group_id * stride_x_row + cols, mask=mask, other=0.0)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row in range(row_start, row_end, G):
        x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0)
        dy = tl.load(DY + row * stride_dy_row + cols, mask=mask, other=0)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w if HAS_WEIGHT else xhat
            if HAS_BIAS:
                y = y + b
            tl.store(Y + row * stride_y_row + cols, y, mask=mask)
        wdy = dy
        if HAS_WEIGHT:
            wdy = dy * w
            dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + row * stride_dres_row + cols, mask=
                mask, other=0)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + row * stride_dres_in_row + cols, dx,
                mask=mask)
        tl.store(DX + row * stride_dx_row + cols, dx, mask=mask)
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)

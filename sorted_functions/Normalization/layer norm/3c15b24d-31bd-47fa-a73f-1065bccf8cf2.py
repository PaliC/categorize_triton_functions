import triton
import triton.language as tl
import torch

@triton.heuristics({'HAS_BIAS': lambda args: args['B'] is not None})
@triton.heuristics({'HAS_Z': lambda args: args['Z'] is not None})
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['Y'] is not None})
@triton.jit
def _layer_norm_bwd_kernel(X, W, B, Z, Y, DY, DX, DW, DB, DZ, Mean, Rstd,
    stride_x_row, stride_z_row, stride_y_row, stride_dy_row, stride_dx_row,
    stride_dz_row, stride_dw_row, stride_db_row, M, N, eps,
    rows_per_program, NORM_BEFORE_GATE: 'tl.constexpr', IS_RMS_NORM:
    'tl.constexpr', HAS_BIAS: 'tl.constexpr', HAS_Z: 'tl.constexpr',
    RECOMPUTE_OUTPUT: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row + group * N
    if HAS_Z:
        Z += row_start * stride_z_row + group * N
        DZ += row_start * stride_dz_row + group * N
    DY += row_start * stride_dy_row + group * N
    DX += row_start * stride_dx_row + group * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    w = tl.load(W + cols, mask=mask)
    if (RECOMPUTE_OUTPUT or HAS_Z) and HAS_BIAS:
        B += group * N
        b = tl.load(B + cols, mask=mask, other=0.0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0)
        dy = tl.load(DY + cols, mask=mask, other=0)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0)
            x_og = x
            x = x_og * z * tl.sigmoid(z)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0)
            z_sigmoid = tl.sigmoid(z)
            y = xhat * w + b if HAS_BIAS else xhat * w
            if RECOMPUTE_OUTPUT:
                tl.store(Y + cols, y * z * z_sigmoid, mask=mask)
            dz = dy * y * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dy *= z * z_sigmoid
        elif RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=0) / N
        if not IS_RMS_NORM:
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            dx = (wdy - xhat * c1) * rstd
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_Z and not NORM_BEFORE_GATE:
            z_sigmoid = tl.sigmoid(z)
            dz = dx * x_og * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dx *= z * z_sigmoid
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_Z:
            Z += stride_z_row
            DZ += stride_dz_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * stride_dw_row + group * N + cols, dw, mask
        =mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * stride_db_row + group * N + cols, db,
            mask=mask)

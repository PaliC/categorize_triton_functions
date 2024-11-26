import triton
import triton.language as tl
import torch

@triton.jit
def _ln_mul_dropout_bwd_dx_du(DX, DU, DY, DW, DB, X, U, Y, W, B, Mean, Rstd,
    stride_dx, stride_du, stride_dy, stride_x, stride_u, stride_y, D, eps,
    seed, dropout_ratio, N, BLOCK_D: 'tl.constexpr', TRAINING:
    'tl.constexpr', CONCAT_UX: 'tl.constexpr', COMPUTE_Y: 'tl.constexpr'):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1
    if rows_per_tile == 0:
        return
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    row = pid
    X += row * stride_x
    U += row * stride_u
    if COMPUTE_Y:
        Y += row * stride_y
    DY += row * stride_dy
    DX += row * stride_dx
    DU += row * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols
    for idx in range(0, rows_per_tile):
        x = tl.load(X + cols, mask=mask, other=0)
        if CONCAT_UX:
            du = tl.load(DY + cols, mask=mask, other=0)
            dx = tl.load(DY + D + cols, mask=mask, other=0)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0)
        if TRAINING:
            random_offsets = row * BLOCK_D + cols
            if CONCAT_UX:
                random_du = tl.rand(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                random_dx = tl.rand(seed, random_offsets + D)
                dx_keep = random_dx > dropout_ratio
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                random_dy = tl.rand(seed, random_offsets + 2 * D)
                dy_keep = random_dy > dropout_ratio
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                random = tl.rand(seed, random_offsets)
                dy_keep = random > dropout_ratio
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        ln = xhat * w + b
        du += dy * ln
        tl.store(DU + cols, du, mask=mask)
        u = tl.load(U + cols, mask=mask, other=0)
        dy = dy * u
        wdy = w * dy
        if COMPUTE_Y:
            y = ln * u
            if TRAINING:
                if CONCAT_UX:
                    u = tl.where(du_keep, u / (1.0 - dropout_ratio), 0.0)
                    x = tl.where(dx_keep, x / (1.0 - dropout_ratio), 0.0)
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
                else:
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
            if CONCAT_UX:
                tl.store(Y + cols, u, mask=mask)
                tl.store(Y + D + cols, x, mask=mask)
                tl.store(Y + 2 * D + cols, y, mask=mask)
            else:
                tl.store(Y + cols, y, mask=mask)
            Y += tile_num * stride_y
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + cols, dx, mask=mask)
        partial_dw = dy * xhat
        partial_db = dy
        if idx > 0:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)
        X += tile_num * stride_x
        U += tile_num * stride_u
        DY += tile_num * stride_dy
        DX += tile_num * stride_dx
        DU += tile_num * stride_du
        row += tile_num

import triton
import triton.language as tl
import torch

@triton.jit
def _group_norm_mul_dropout_bwd_dx_du(DX, DU, DY, DW, DB, X, U, Y, W, B,
    Mean, Rstd, stride_dx, stride_du, stride_dy, stride_x, stride_u,
    stride_y, D, Heads, eps, seed, dropout_ratio, GROUP_N: 'tl.constexpr',
    BLOCK_D: 'tl.constexpr', BLOCK_H: 'tl.constexpr', TRAINING:
    'tl.constexpr', CONCAT_UX: 'tl.constexpr', COMPUTE_Y: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    off_heads = tl.arange(0, BLOCK_H)
    mask_c = cols < D
    mask_h = off_heads < Heads
    mask = mask_c[None, :] & mask_h[:, None]
    X += row * stride_x
    U += row * stride_u
    DY += row * stride_dy
    DX += row * stride_dx
    DU += row * stride_du
    offsets = off_heads[:, None] * D + cols[None, :]
    x = tl.load(X + offsets, mask=mask, other=0)
    if CONCAT_UX:
        du = tl.load(DY + offsets, mask=mask, other=0)
        dx = tl.load(DY + Heads * D + offsets, mask=mask, other=0)
        dy = tl.load(DY + 2 * Heads * D + offsets, mask=mask, other=0)
    else:
        du = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dx = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dy = tl.load(DY + offsets, mask=mask, other=0)
    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            random_du = tl.rand(seed, random_offsets)
            du_keep = random_du > dropout_ratio
            du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
            random_dx = tl.rand(seed, random_offsets + Heads * D)
            dx_keep = random_dx > dropout_ratio
            dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
            random_dy = tl.rand(seed, random_offsets + 2 * Heads * D)
            dy_keep = random_dy > dropout_ratio
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            dy_keep = random > dropout_ratio
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
    mean = tl.load(Mean + row * Heads + off_heads)
    rstd = tl.load(Rstd + row * Heads + off_heads)
    xhat = (x - mean[:, None]) * rstd[:, None]
    w = tl.load(W + off_heads, mask=mask_h)
    b = tl.load(B + off_heads, mask=mask_h)
    ln = xhat * w[:, None] + b[:, None]
    du += dy * ln
    tl.store(DU + offsets, du, mask=mask)
    u = tl.load(U + offsets, mask=mask, other=0)
    dy = dy * u
    wdy = w[:, None] * dy
    if COMPUTE_Y:
        Y += row * stride_y
        y = ln * u
        if TRAINING:
            if CONCAT_UX:
                u = tl.where(du_keep, u / (1.0 - dropout_ratio), 0.0)
                x = tl.where(dx_keep, x / (1.0 - dropout_ratio), 0.0)
                y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
            else:
                y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
        if CONCAT_UX:
            tl.store(Y + offsets, u, mask=mask)
            tl.store(Y + Heads * D + offsets, x, mask=mask)
            tl.store(Y + 2 * Heads * D + offsets, y, mask=mask)
        else:
            tl.store(Y + offsets, y, mask=mask)
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=1) / D
    c2 = tl.sum(wdy, axis=1) / D
    dx += (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    tl.store(DX + offsets, dx, mask=mask)
    lock_id = row % GROUP_N
    DW = DW + lock_id * Heads + off_heads
    DB = DB + lock_id * Heads + off_heads
    partial_dw = tl.sum(dy * xhat, axis=1)
    partial_dw = tl.ravel(partial_dw)
    partial_db = tl.sum(dy, axis=1)
    partial_db = tl.ravel(partial_db)
    tl.atomic_add(DW, partial_dw, mask=mask_h, sem='relaxed')
    tl.atomic_add(DB, partial_db, mask=mask_h, sem='relaxed')

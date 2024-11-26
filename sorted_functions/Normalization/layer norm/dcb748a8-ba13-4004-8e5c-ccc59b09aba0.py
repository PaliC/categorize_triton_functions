import triton
import triton.language as tl
import torch

@triton.jit
def _weighted_layer_norm_bwd_dx(DX, DY, DW, DB, X, W, B, Mean, Rstd,
    stride_dx, stride_dy, stride_x, D, eps, IS_SWISH: 'tl.constexpr', N,
    BLOCK_D: 'tl.constexpr'):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    row = pid
    for idx in range(rows_per_tile):
        x_ptrs = X + row * stride_x
        dy_ptrs = DY + row * stride_dy
        dx_ptrs = DX + row * stride_dx
        dw_ptrs = DW + pid * D
        dw_ptrs += cols
        db_ptrs = DB + pid * D
        db_ptrs += cols
        x = tl.load(x_ptrs + cols, mask=mask, other=0)
        dy = tl.load(dy_ptrs + cols, mask=mask, other=0)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        w = tl.load(W + cols, mask=mask)
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        sigmoid_layer_norm = None
        if IS_SWISH:
            b = tl.load(B + cols, mask=mask)
            sigmoid_layer_norm = tl.sigmoid(xhat * w + b)
            sigmoid_layer_norm = tl.where(mask, sigmoid_layer_norm, 0.0)
            x_ = wdy * x * sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            x_ = tl.where(mask, x_, 0.0)
            c1 = tl.sum(xhat * x_, axis=0) / D
            c2 = tl.sum(x_, axis=0) / D
            dx = (x_ - (xhat * c1 + c2)) * rstd
            dx = dy * sigmoid_layer_norm + dx
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / D
            c2 = tl.sum(wdy, axis=0) / D
            dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(dx_ptrs + cols, dx, mask=mask)
        if IS_SWISH:
            partial_dw = dy * x * xhat * sigmoid_layer_norm * (1 -
                sigmoid_layer_norm)
            partial_db = dy * x * sigmoid_layer_norm * (1 - sigmoid_layer_norm)
        else:
            partial_dw = dy * xhat
            partial_db = dy
        if idx > 0:
            partial_dw += tl.load(dw_ptrs, mask=mask)
            partial_db += tl.load(db_ptrs, mask=mask)
        tl.store(dw_ptrs, partial_dw, mask=mask)
        tl.store(db_ptrs, partial_db, mask=mask)
        row += tile_num

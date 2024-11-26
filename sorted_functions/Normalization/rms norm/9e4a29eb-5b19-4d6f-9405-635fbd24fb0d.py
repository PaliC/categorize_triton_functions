import triton
import triton.language as tl
import torch

@triton.jit
def _rmsnorm_bwd_kernel(X, W, DY, DX, DW, Rstd, stride_x_row, stride_dy_row,
    stride_dx_row, M, N, eps, rows_per_program, BLOCK_N: 'tl.constexpr',
    IS_EVEN_N: 'tl.constexpr'):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    w = tl.load(W + cols, mask=mask)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        if IS_EVEN_N:
            x = tl.load(X + cols)
            dy = tl.load(DY + cols)
        else:
            x = tl.load(X + cols, mask=mask, other=0)
            dy = tl.load(DY + cols, mask=mask, other=0)
        rstd = tl.load(Rstd + row)
        xhat = x * rstd
        if not IS_EVEN_N:
            xhat = tl.where(mask, xhat, 0.0)
        wdy = w * dy
        dw += dy * xhat
        c1 = tl.sum(xhat * wdy, axis=0) / N
        dx = (wdy - xhat * c1) * rstd
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)

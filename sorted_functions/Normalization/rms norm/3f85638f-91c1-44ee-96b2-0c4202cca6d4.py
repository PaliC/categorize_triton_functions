import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N'])
@triton.jit
def _rms_norm_bwd_kernel_sm(X, stride_x, W, DY, stride_dy, DX, stride_dx,
    Rstd, DW, eps, M, N, rows_per_program, block_N: 'tl.constexpr'):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, block_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0)
    dw = tl.zeros((block_N,), dtype=tl.float32)
    row_end = min(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0)
        rstd = tl.load(Rstd + row)
        x_hat = x * rstd
        wdy = w * dy
        dw += dy * x_hat
        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = (wdy - x_hat * c1) * rstd
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)

import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_N': 32}), triton.Config({
    'BLOCK_N': 64}), triton.Config({'BLOCK_N': 128}), triton.Config({
    'BLOCK_N': 256}), triton.Config({'BLOCK_N': 512}), triton.Config({
    'BLOCK_N': 1024})], key=['ncols'])
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['OUT'] is not None})
@triton.jit
def _swiglu_bwd_kernel(X, Y, DOUT, OUT, DX, DY, stride_x_row, stride_y_row,
    stride_dout_row, stride_out_row, stride_dx_row, stride_dy_row, ncols,
    BLOCK_N: 'tl.constexpr', RECOMPUTE_OUTPUT: 'tl.constexpr'):
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    DOUT += row * stride_dout_row
    if RECOMPUTE_OUTPUT:
        OUT += row * stride_out_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0)
    dout = tl.load(DOUT + cols, mask=cols < ncols, other=0.0)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < ncols)
    tl.store(DY + cols, dy, mask=cols < ncols)
    if RECOMPUTE_OUTPUT:
        out = x * x_sigmoid * y
        tl.store(OUT + cols, out, mask=cols < ncols)

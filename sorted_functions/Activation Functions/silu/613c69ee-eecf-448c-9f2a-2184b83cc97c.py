import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_N': 32}), triton.Config({
    'BLOCK_N': 64}), triton.Config({'BLOCK_N': 128}), triton.Config({
    'BLOCK_N': 256}), triton.Config({'BLOCK_N': 512}), triton.Config({
    'BLOCK_N': 1024})], key=['ncols'])
@triton.jit
def _swiglu_fwd_kernel(X, Y, OUT, stride_x_row, stride_y_row,
    stride_out_row, ncols, BLOCK_N: 'tl.constexpr'):
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    OUT += row * stride_out_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)

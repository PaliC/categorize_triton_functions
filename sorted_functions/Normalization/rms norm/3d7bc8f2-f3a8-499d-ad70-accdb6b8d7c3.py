import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}
    ), triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8}), triton.Config({
    'BLOCK_SIZE': 512, 'NUM_WARPS': 16}), triton.Config({'BLOCK_SIZE': 1024,
    'NUM_WARPS': 16}), triton.Config({'BLOCK_SIZE': 2048, 'NUM_WARPS': 32}),
    triton.Config({'BLOCK_SIZE': 4096, 'NUM_WARPS': 32}), triton.Config({
    'BLOCK_SIZE': 8192, 'NUM_WARPS': 48})], key=['n_cols'])
@triton.jit
def _rms_layernorm_backward(dY, dY_row_stride, X, X_row_stride, W,
    W_row_stride, r, r_row_stride, dX, dX_row_stride, dW, n_cols, eps,
    BLOCK_SIZE: 'tl.constexpr', NUM_WARPS: 'tl.constexpr'):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY_ptr = dY + pid * dY_row_stride + col_offsets
    X_ptr = X + pid * X_row_stride + col_offsets
    dX_ptr = dX + pid * dX_row_stride + col_offsets
    dY_row = tl.load(dY_ptr, mask=mask, other=0)
    X_row = tl.load(X_ptr, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    rms = tl.load(r + pid)
    X_norm = X_row * rms
    dY_W = dY_row * W_row
    sum_dY_X = tl.sum(dY_W * X_norm, axis=0)
    dX = rms * (dY_W - X_norm * (sum_dY_X / n_cols))
    dW_row = dY_row * X_norm
    tl.atomic_add(dW + col_offsets, dW_row, mask=mask)
    tl.store(dX_ptr, dX, mask=mask)

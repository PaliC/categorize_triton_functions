import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_fwd(X, Y, W, Rstd, N, eps, BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    row_offset = tl.program_id(0) * BLOCK_M
    row_index = row_offset + tl.arange(0, BLOCK_M)[:, None]
    col_index = tl.arange(0, BLOCK_N)[None, :]
    col_mask = col_index < N
    x = tl.load(X + N * row_index + col_index, col_mask, other=0.0)
    w = tl.load(W + col_index, col_mask, eviction_policy='evict_last',
        other=0.0)
    xx = x * x
    xx = tl.broadcast_to(xx, [BLOCK_M, BLOCK_N])
    mean = tl.sum(xx, axis=1)[:, None] / N
    rstd = tl.rsqrt(mean + eps)
    y = x * rstd * w
    tl.store(Rstd + row_index, rstd)
    tl.store(Y + N * row_index + col_index, y, col_mask)

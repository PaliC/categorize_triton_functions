import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_add_kernel(x_ptr, y_ptr, h1_ptr, w_ptr, eps, stride, N_COLS:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', INCLUDE_WEIGHT: 'tl.constexpr'
    ):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    h1_ptr += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        ax = tl.load(x_ptr + cols, mask=mask, other=0.0, eviction_policy=
            'evict_last')
        ay = tl.load(y_ptr + cols, mask=mask, other=0.0, eviction_policy=
            'evict_first')
        a = ax + ay
        tl.store(x_ptr + cols, a, mask=mask)
        _mean += a * a
    rstd = rsqrt(tl.sum(_mean, axis=0) / N_COLS + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(x_ptr + cols, mask=mask, other=0.0, eviction_policy=
            'evict_first')
        if INCLUDE_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(h1_ptr + cols, a * rstd * w, mask=mask)
        else:
            tl.store(h1_ptr + cols, a * rstd, mask=mask)

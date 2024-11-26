import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_A': 4}, num_warps=1),
    triton.Config({'BLOCK_A': 16}, num_warps=2), triton.Config({'BLOCK_A': 
    32}, num_warps=4), triton.Config({'BLOCK_A': 64}, num_warps=8), triton.
    Config({'BLOCK_A': 128}, num_warps=16), triton.Config({'BLOCK_A': 256},
    num_warps=32), triton.Config({'BLOCK_A': 8}, num_warps=1), triton.
    Config({'BLOCK_A': 16}, num_warps=2), triton.Config({'BLOCK_A': 32},
    num_warps=4), triton.Config({'BLOCK_A': 64}, num_warps=8), triton.
    Config({'BLOCK_A': 128}, num_warps=16), triton.Config({'BLOCK_A': 256},
    num_warps=32), triton.Config({'BLOCK_A': 16}, num_warps=1), triton.
    Config({'BLOCK_A': 32}, num_warps=2), triton.Config({'BLOCK_A': 64},
    num_warps=4), triton.Config({'BLOCK_A': 128}, num_warps=8), triton.
    Config({'BLOCK_A': 256}, num_warps=16), triton.Config({'BLOCK_A': 512},
    num_warps=32)], key=['A', 'MAX_INTERP'])
@triton.jit
def __scan_col_compute(X, stride_xn, stride_xa, stride_xb, N, A, B:
    'tl.constexpr', SCALE, stride_scale, NCOLS, stride_ncolsn,
    stride_ncolsa, COL_INDICES, stride_coln, stride_cola, stride_colz,
    MAX_Z, MAX_INTERP: 'tl.constexpr', ORIGINAL_WIDTH: 'tl.constexpr',
    TARGET_WIDTH_MAX: 'tl.constexpr', BLOCK_A: 'tl.constexpr'):
    n = tl.program_id(0)
    pid_a = tl.program_id(1)
    index_as = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    mask_as = index_as < A
    scales_as = tl.load(SCALE + index_as * stride_scale, mask=mask_as, other=0)
    last_index = tl.zeros((BLOCK_A,), dtype=tl.int32)
    for _b in range(B):
        b = _b % ORIGINAL_WIDTH
        x_mask = tl.load(X + n * stride_xn + index_as * stride_xa + _b *
            stride_xb, mask=mask_as, other=0)
        v_start = tl.math.round(b * scales_as)
        v_end = tl.math.round((b + 1) * scales_as)
        n_pixel = (v_end - v_start) * x_mask
        tl.store(COL_INDICES + n * stride_coln + index_as[:, None] *
            stride_cola + (tl.arange(0, MAX_INTERP)[None, :] + last_index[:,
            None]) * stride_colz, tl.arange(0, MAX_INTERP)[None, :] +
            v_start[:, None] + tl.math.floor(tl.math.floor(_b /
            ORIGINAL_WIDTH) * TARGET_WIDTH_MAX), mask=(tl.arange(0,
            MAX_INTERP)[None, :] < n_pixel[:, None]) & mask_as[:, None])
        last_index += n_pixel
    tl.store(NCOLS + n * stride_ncolsn + index_as * stride_ncolsa,
        last_index, mask=mask_as)

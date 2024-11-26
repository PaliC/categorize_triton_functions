import triton
import triton.language as tl
import torch

@triton.jit
def __scan_col_compute_old(X, stride_xn, stride_xa, stride_xb, N, A, B:
    'tl.constexpr', BLOCK_A: 'tl.constexpr', SCALE, stride_scale, NCOLS,
    stride_ncolsn, stride_ncolsa, COL_INDICES, stride_coln, stride_cola,
    stride_colz, MAX_Z: 'tl.constexpr', MAX_INTERP: 'tl.constexpr',
    ORIGINAL_WIDTH: 'tl.constexpr', TARGET_WIDTH_MAX: 'tl.constexpr',
    GRID_N, GRID_A):
    n = tl.program_id(0)
    pid_a = tl.program_id(1)
    for ia in range(BLOCK_A):
        a = ia * GRID_A + pid_a
        mask_a = a < A
        scales_a = tl.load(SCALE + a * stride_scale, mask=mask_a, other=0)
        last_index = int(0)
        for _b in range(B):
            b = _b % ORIGINAL_WIDTH
            x_mask = tl.load(X + n * stride_xn + a * stride_xa + _b *
                stride_xb, mask=mask_a, other=0)
            v_start = tl.math.round(b * scales_a)
            v_end = tl.math.round((b + 1) * scales_a)
            n_pixel = (v_end - v_start) * x_mask
            tl.store(COL_INDICES + n * stride_coln + a * stride_cola + (tl.
                arange(0, MAX_INTERP) + last_index) * stride_colz, tl.
                arange(0, MAX_INTERP) + v_start + tl.math.floor(tl.math.
                floor(_b / ORIGINAL_WIDTH) * TARGET_WIDTH_MAX), mask=(tl.
                arange(0, MAX_INTERP) < n_pixel) & mask_a)
            last_index += n_pixel
        tl.store(NCOLS + n * stride_ncolsn + a * stride_ncolsa, last_index,
            mask=mask_a)

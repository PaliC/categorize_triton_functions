import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['BLOCK_N_ZERO', 'BLOCK_ROW', 'MAX_INTERP'])
@triton.jit
def __scan_col_3_compute(NON_ZERO_ROWS, stride_nzr_n, stride_nzr_d,
    PIXEL_INDICES, stride_pixel_n, stride_pixel_m, V_STARTS, stride_vs_tdst,
    stride_vs_tm, COL_INDICES, stride_col_n, stride_col_z, N, M, H, T_M,
    TARGET_WIDTH_MAX: 'tl.constexpr', MAX_INTERP: 'tl.constexpr', NZR_N,
    NZR_D, BLOCK_N_ZERO: 'tl.constexpr', NCOL_PER_ROW, BLOCK_ROW:
    'tl.constexpr'):
    pid_nzr = tl.program_id(0)
    pid_col = tl.program_id(1)
    for _i_nzr in range(BLOCK_N_ZERO):
        i_nzr = pid_nzr * BLOCK_N_ZERO + _i_nzr
        mask_nzr = i_nzr < NZR_N
        i_batch = tl.load(NON_ZERO_ROWS + i_nzr * stride_nzr_n + 0 *
            stride_nzr_d, mask=mask_nzr)
        i_row = tl.load(NON_ZERO_ROWS + i_nzr * stride_nzr_n + 1 *
            stride_nzr_d, mask=mask_nzr)
        n = i_batch
        ms = pid_col * BLOCK_ROW + tl.arange(0, BLOCK_ROW
            ) + i_row * NCOL_PER_ROW
        ms_mask = pid_col * BLOCK_ROW + tl.arange(0, BLOCK_ROW) < NCOL_PER_ROW
        idx_tdst = ms // (H * T_M)
        idx_h = ms % (H * T_M) // T_M
        idx_tm = ms % T_M
        v_start = tl.load(V_STARTS + idx_tdst * stride_vs_tdst + idx_tm *
            stride_vs_tm, mask=ms_mask)
        col_start = tl.load(PIXEL_INDICES + n * stride_pixel_n + (ms - 1) *
            stride_pixel_m, mask=(ms - 1 >= 0 and ms < M) and ms_mask)
        col_end = tl.load(PIXEL_INDICES + n * stride_pixel_n + ms *
            stride_pixel_m, mask=(ms >= 0 and ms < M) and ms_mask)
        col_len = col_end - col_start
        range_start = v_start + idx_h * TARGET_WIDTH_MAX
        tl.store(COL_INDICES + n * stride_col_n + (tl.arange(0, MAX_INTERP)
            [None, :] + col_start[:, None]) * stride_col_z, tl.arange(0,
            MAX_INTERP)[None, :] + range_start[:, None], mask=tl.arange(0,
            MAX_INTERP)[None, :] < col_len[:, None] and ms_mask[:, None])

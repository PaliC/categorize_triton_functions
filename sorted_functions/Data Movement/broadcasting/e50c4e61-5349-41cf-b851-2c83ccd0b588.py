import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['BLOCK_M', 'BLOCK_M', 'MAX_INTERP'])
@triton.jit
def __scan_col_2_compute(PIXEL_INDICES, stride_pixel_n, stride_pixel_m,
    V_STARTS, stride_vs_tdst, stride_vs_tm, COL_INDICES, stride_col_n,
    stride_col_z, N, M, H, T_M, TARGET_WIDTH_MAX, BLOCK_N: 'tl.constexpr',
    GROUP_M: 'tl.constexpr', BLOCK_M: 'tl.constexpr', MAX_INTERP:
    'tl.constexpr'):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    grid_m = tl.program_id(1)
    for _n in range(BLOCK_N):
        for _m in range(0, GROUP_M):
            n = pid_n * BLOCK_N + _n
            ms = pid_m * BLOCK_M * GROUP_M + _m * BLOCK_M + tl.arange(0,
                BLOCK_M)
            ms_mask = ms < M
            idx_tdst = ms // (H * T_M)
            idx_h = ms % (H * T_M) // T_M
            idx_tm = ms % T_M
            v_start = tl.load(V_STARTS + idx_tdst * stride_vs_tdst + idx_tm *
                stride_vs_tm, mask=ms_mask)
            col_start = tl.load(PIXEL_INDICES + n * stride_pixel_n + (ms - 
                1) * stride_pixel_m, mask=(ms - 1 >= 0 and ms < M) and ms_mask)
            col_end = tl.load(PIXEL_INDICES + n * stride_pixel_n + ms *
                stride_pixel_m, mask=(ms >= 0 and ms < M) and ms_mask)
            col_len = col_end - col_start
            range_start = v_start + idx_h * TARGET_WIDTH_MAX
            tl.store(COL_INDICES + n * stride_col_n + (tl.arange(0,
                MAX_INTERP)[None, :] + col_start[:, None]) * stride_col_z, 
                tl.arange(0, MAX_INTERP)[None, :] + range_start[:, None],
                mask=tl.arange(0, MAX_INTERP)[None, :] < col_len[:, None] and
                ms_mask[:, None])

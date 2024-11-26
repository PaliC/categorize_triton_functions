import triton
import triton.language as tl
import torch

@triton.jit
def __scan_col_4_compute(NON_ZERO_PIXELS, stride_nzp_n, stride_nzp_d,
    PIXEL_INDICES, stride_pixel_n, stride_pixel_m, V_STARTS, stride_vs_tdst,
    stride_vs_tm, V_ENDS, stride_ve_tdst, stride_ve_tm, COL_INDICES,
    stride_col_n, stride_col_z, N, M, H, T_M, TARGET_WIDTH_MAX,
    MAX_INTER_PADDED: 'tl.constexpr', MAX_INTERP, NZR_N, NZR_D,
    BLOCK_N_ZERO: 'tl.constexpr'):
    pid_nzp = tl.program_id(0)
    i_nzp_n = pid_nzp * BLOCK_N_ZERO + tl.arange(0, BLOCK_N_ZERO)
    mask_i_nzp = i_nzp_n < NZR_N
    is_batch = tl.load(NON_ZERO_PIXELS + i_nzp_n * stride_nzp_n + 0 *
        stride_nzp_d, mask=mask_i_nzp)
    is_col = tl.load(NON_ZERO_PIXELS + i_nzp_n * stride_nzp_n + 1 *
        stride_nzp_d, mask=mask_i_nzp)
    idx_tdst = is_col // (H * T_M)
    idx_h = is_col % (H * T_M) // T_M
    idx_tm = is_col % T_M
    v_start = tl.load(V_STARTS + idx_tdst * stride_vs_tdst + idx_tm *
        stride_vs_tm, mask=mask_i_nzp)
    v_end = tl.load(V_ENDS + idx_tdst * stride_ve_tdst + idx_tm *
        stride_ve_tm, mask=mask_i_nzp)
    col_start = tl.load(PIXEL_INDICES + is_batch * stride_pixel_n + (is_col -
        1) * stride_pixel_m, mask=(is_col - 1 >= 0 and is_col < M) and
        mask_i_nzp, other=0)
    col_end = tl.load(PIXEL_INDICES + is_batch * stride_pixel_n + is_col *
        stride_pixel_m, mask=(is_col >= 0 and is_col < M) and mask_i_nzp)
    col_len = col_end - col_start
    range_start = v_start + idx_h * TARGET_WIDTH_MAX
    range_end = v_end + idx_h * TARGET_WIDTH_MAX
    tl.store(COL_INDICES + is_batch[:, None] * stride_col_n + (tl.arange(0,
        MAX_INTER_PADDED)[None, :] + col_start[:, None]) * stride_col_z, 
        range_end[:, None] - tl.arange(0, MAX_INTER_PADDED)[None, :] * ((
        range_end[:, None] - range_start[:, None]) / col_len[:, None]) - 1,
        mask=(tl.arange(0, MAX_INTER_PADDED)[None, :] < col_len[:, None] and
        tl.arange(0, MAX_INTER_PADDED)[None, :] < MAX_INTERP) and
        mask_i_nzp[:, None])

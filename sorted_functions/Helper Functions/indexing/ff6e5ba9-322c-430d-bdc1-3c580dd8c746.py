import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['BLOCK_A', 'MAX_NCOLS'])
@triton.jit
def __compact_cols_compute(NCOLS_CS, stride_ncols_cs_n, stride_ncols_cs_a,
    N, A, COL_INDICES, stride_col_indices_n, stride_col_indices_a,
    stride_col_indices_mz, OUT_COL_INDICES, stride_out_col_indices_n,
    stride_out_col_indices_z, MAX_NCOLS: 'tl.constexpr', BLOCK_A:
    'tl.constexpr'):
    n = tl.program_id(0)
    pid_a = tl.program_id(1)
    for ia in range(BLOCK_A):
        a = pid_a * BLOCK_A + ia
        mask_a = a < A
        cs_start = tl.load(NCOLS_CS + n * stride_ncols_cs_n + a *
            stride_ncols_cs_a, mask=mask_a)
        cs_end = tl.load(NCOLS_CS + n * stride_ncols_cs_n + (a + 1) *
            stride_ncols_cs_a, mask=mask_a)
        cs_len = cs_end - cs_start
        col_indices = tl.load(COL_INDICES + n * stride_col_indices_n + a *
            stride_col_indices_a + tl.arange(0, MAX_NCOLS), mask=(tl.arange
            (0, MAX_NCOLS) < cs_len) & mask_a)
        tl.store(OUT_COL_INDICES + n * stride_out_col_indices_n + (tl.
            arange(0, MAX_NCOLS) + cs_start) * stride_out_col_indices_z,
            col_indices, mask=(tl.arange(0, MAX_NCOLS) < cs_len) & mask_a)

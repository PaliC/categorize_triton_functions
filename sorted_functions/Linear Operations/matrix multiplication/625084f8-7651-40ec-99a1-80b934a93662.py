import triton
import triton.language as tl
import torch

@triton.jit
def __flat_csr_elmul_compute(CROW_INDICES, stride_crow_n, stride_crow_r,
    COL_INDICES, stride_col_n, stride_col_z, IN_VALUES, stride_in_n,
    stride_in_z, OUT_VALUES, stride_out_n, stride_out_z, OTHER,
    stride_other_n, stride_other_h, stride_other_tdst, stride_other_tsrc, N,
    H, T_DST, T_SRC, R, Z, MAX_ROW_Z: 'tl.constexpr', BLOCK_R: 'tl.constexpr'):
    n = tl.program_id(0)
    ir = tl.program_id(1)
    ir = ir * BLOCK_R + tl.arange(0, BLOCK_R)
    ir_mask = ir < R
    crow_start = tl.load(CROW_INDICES + n * stride_crow_n + ir *
        stride_crow_r, mask=ir_mask)
    crow_end = tl.load(CROW_INDICES + n * stride_crow_n + (ir + 1) *
        stride_crow_r, mask=ir_mask)
    idx_ht = tl.load(COL_INDICES + n * stride_col_n + (tl.arange(0,
        MAX_ROW_Z)[None, :] + crow_start[:, None]) * stride_col_z, mask=tl.
        arange(0, MAX_ROW_Z)[None, :] < crow_end[:, None] - crow_start[:,
        None] and ir_mask[:, None])
    idx_heads = idx_ht // T_SRC
    idx_cols = idx_ht % T_SRC
    in_values = tl.load(IN_VALUES + n * stride_in_n + (tl.arange(0,
        MAX_ROW_Z)[None, :] + crow_start[:, None]) * stride_in_z, mask=tl.
        arange(0, MAX_ROW_Z)[None, :] < crow_end[:, None] - crow_start[:,
        None] and ir_mask[:, None])
    other_values = tl.load(OTHER + n * stride_other_n + idx_heads *
        stride_other_h + ir[:, None] * stride_other_tdst + idx_cols *
        stride_other_tsrc, mask=tl.arange(0, MAX_ROW_Z)[None, :] < crow_end
        [:, None] - crow_start[:, None] and ir_mask[:, None])
    out_values = in_values * other_values
    tl.store(OUT_VALUES + n * stride_out_n + (tl.arange(0, MAX_ROW_Z)[None,
        :] + crow_start[:, None]) * stride_out_z, out_values, mask=tl.
        arange(0, MAX_ROW_Z)[None, :] < crow_end[:, None] - crow_start[:,
        None] and ir_mask[:, None])

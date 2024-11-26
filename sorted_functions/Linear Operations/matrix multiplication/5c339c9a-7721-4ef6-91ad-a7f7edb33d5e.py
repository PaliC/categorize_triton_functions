import triton
import triton.language as tl
import torch

@triton.jit
def __flat_csr_sdbmm_tch_compute(CROW_INDICES, stride_crow_n, stride_crow_r,
    COL_INDICES, stride_col_n, stride_col_z, VALUES, stride_v_n, stride_v_z,
    OTHER, stride_other_n, stride_other_h, stride_other_t, stride_other_d,
    OUTPUT, stride_output_n, stride_output_h, stride_output_t,
    stride_output_d, TEMP_COUNT_HEAD, stride_tch_n, stride_tch_r,
    stride_tch_h, N, R, Z, H, T_DST, T_SRC, HID, MAX_ROW_Z: 'tl.constexpr',
    MAX_ROW_T: 'tl.constexpr', BLOCK_HID: 'tl.constexpr', BLOCK_H:
    'tl.constexpr', BLOCK_R: 'tl.constexpr', BLOCK_COL_HEAD: 'tl.constexpr',
    GRID_COL_HEAD: 'tl.constexpr'):
    n = tl.program_id(0)
    pid_ir = tl.program_id(1)
    grid_ir = tl.num_programs(1)
    for _ir in range(BLOCK_R):
        ir = _ir * grid_ir + pid_ir
        ir_mask = ir < R
        crow_start = tl.load(CROW_INDICES + n * stride_crow_n + ir *
            stride_crow_r, mask=ir_mask)
        crow_end = tl.load(CROW_INDICES + n * stride_crow_n + (ir + 1) *
            stride_crow_r, mask=ir_mask)
        count_heads_sum = tl.zeros((BLOCK_H,), dtype=tl.int32)
        for i in range(GRID_COL_HEAD):
            _col_indices = tl.load(COL_INDICES + n * stride_col_n + (tl.
                arange(0, BLOCK_COL_HEAD) + crow_start + i * BLOCK_COL_HEAD
                ) * stride_col_z, mask=(tl.arange(0, BLOCK_COL_HEAD) +
                crow_start + i * BLOCK_COL_HEAD < crow_end) & ir_mask,
                other=T_SRC * BLOCK_H * 2)
            t = _col_indices // T_SRC
            count_heads_sum += tl.sum(t[None, :] == tl.arange(0, BLOCK_H)[:,
                None], axis=1)
        count_heads_cumsum = tl.cumsum(count_heads_sum)
        tl.store(TEMP_COUNT_HEAD + n * stride_tch_n + ir * stride_tch_r + 
            tl.arange(0, BLOCK_H) * stride_tch_h, value=count_heads_cumsum,
            mask=(tl.arange(0, BLOCK_H) < H) & ir_mask)

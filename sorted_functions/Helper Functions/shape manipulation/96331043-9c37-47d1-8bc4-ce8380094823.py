import triton
import triton.language as tl
import torch

@triton.jit
def __flat_csr_sdbmm_compute(CROW_INDICES, stride_crow_n, stride_crow_r,
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
    pid_hid = tl.program_id(2)
    for _ir in range(BLOCK_R):
        ir = _ir * grid_ir + pid_ir
        ir_mask = ir < R
        crow_start = tl.load(CROW_INDICES + n * stride_crow_n + ir *
            stride_crow_r, mask=ir_mask)
        crow_end = tl.load(CROW_INDICES + n * stride_crow_n + (ir + 1) *
            stride_crow_r, mask=ir_mask)
        for ih in range(H):
            ch_start = tl.load(TEMP_COUNT_HEAD + n * stride_tch_n + ir *
                stride_tch_r + (ih - 1) * stride_tch_h, mask=(ih - 1 >= 0) &
                (ih - 1 < H) & ir_mask, other=0)
            ch_end = tl.load(TEMP_COUNT_HEAD + n * stride_tch_n + ir *
                stride_tch_r + ih * stride_tch_h, mask=(ih < H) & ir_mask,
                other=-1)
            ch_len = ch_end - ch_start
            per_head_col_indices_mask = tl.arange(0, MAX_ROW_T) < ch_len
            per_head_col_indices = tl.load(COL_INDICES + n * stride_col_n +
                (tl.arange(0, MAX_ROW_T) + ch_start + crow_start) *
                stride_col_z, mask=per_head_col_indices_mask & ir_mask, other=1
                ) % T_SRC
            row_values = tl.load(VALUES + n * stride_v_n + (tl.arange(0,
                MAX_ROW_T) + ch_start + crow_start) * stride_v_z, mask=
                per_head_col_indices_mask & ir_mask, other=0)
            hid_range = tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID
            hid_mask = hid_range < HID
            other_mask = per_head_col_indices_mask[:, None] & hid_mask[None, :
                ] & ir_mask
            other = tl.load(OTHER + n * stride_other_n + ih *
                stride_other_h + per_head_col_indices[:, None] *
                stride_other_t + hid_range[None, :] * stride_other_d, mask=
                other_mask, other=0)
            tl.static_assert(other.shape[0] == MAX_ROW_T)
            tl.static_assert(other.shape[1] == BLOCK_HID)
            other_mul = row_values[:, None] * other
            other_sum = tl.sum(other_mul, axis=0)
            tl.store(OUTPUT + n * stride_output_n + ih * stride_output_h + 
                ir * stride_output_t + (tl.arange(0, BLOCK_HID) + pid_hid *
                BLOCK_HID) * stride_output_d, other_sum, mask=(tl.arange(0,
                BLOCK_HID) + pid_hid * BLOCK_HID < HID) & ir_mask)

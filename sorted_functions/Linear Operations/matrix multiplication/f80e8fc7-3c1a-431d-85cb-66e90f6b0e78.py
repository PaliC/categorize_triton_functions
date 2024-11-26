import triton
import triton.language as tl
import torch

@triton.jit
def __flat_csr_masked_bmm_compute(CROW_INDICES, stride_crow_n,
    stride_crow_r1, COL_INDICES, stride_col_n, stride_col_z, A, stride_a_n,
    stride_a_h, stride_a_t, stride_a_d, B, stride_b_n, stride_b_h,
    stride_b_t, stride_b_d, OUT_VALUES, stride_out_n, stride_out_z, N, R,
    T_SRC, HID, GRID_ROW, GRID_COL, BLOCK_ROW: 'tl.constexpr', BLOCK_COL:
    'tl.constexpr', BLOCK_HID: 'tl.constexpr'):
    n = tl.program_id(0)
    pid_ir = tl.program_id(1)
    pid_icol = tl.program_id(2)
    for _ir in range(BLOCK_ROW):
        ir = _ir * GRID_ROW + pid_ir
        ir_mask = ir < R
        crow_start = tl.load(CROW_INDICES + n * stride_crow_n + ir *
            stride_crow_r1, mask=ir_mask)
        crow_end = tl.load(CROW_INDICES + n * stride_crow_n + (ir + 1) *
            stride_crow_r1, mask=ir_mask)
        index_row = ir
        for ic in range(BLOCK_COL):
            icol = ic + pid_icol * BLOCK_COL + crow_start
            _index_col = tl.load(COL_INDICES + n * stride_col_n + icol *
                stride_col_z, mask=(icol < crow_end) & ir_mask)
            index_col = _index_col % T_SRC
            index_head = _index_col // T_SRC
            accumulator = 0.0
            for ih in range(0, tl.cdiv(HID, BLOCK_HID)):
                index_hids = tl.arange(0, BLOCK_HID) + ih * BLOCK_HID
                index_hids_mask = index_hids < HID
                a_vec = tl.load(A + n * stride_a_n + index_head *
                    stride_a_h + index_row * stride_a_t + index_hids *
                    stride_a_d, mask=index_hids_mask & ir_mask, other=0)
                b_vec = tl.load(B + n * stride_b_n + index_head *
                    stride_b_h + index_col * stride_b_t + index_hids *
                    stride_b_d, mask=index_hids_mask & ir_mask, other=0)
                t = tl.sum(a_vec * b_vec)
                accumulator += t
            tl.store(OUT_VALUES + n * stride_out_n + icol * stride_out_z,
                accumulator, mask=(icol < crow_end) & ir_mask)

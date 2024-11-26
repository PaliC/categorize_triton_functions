import triton
import triton.language as tl
import torch

@triton.jit
def __flat_csr_softmax_compute(CROW_INDICES, stride_crow_n, stride_crow_r,
    COL_INDICES, stride_col_n, stride_col_z, IN_VALUES, stride_in_n,
    stride_in_z, OUT_VALUES, stride_out_n, stride_out_z, N, R, H, T_SRC,
    BLOCK_Z: 'tl.constexpr', BLOCK_R: 'tl.constexpr'):
    n = tl.program_id(0)
    pid_ir = tl.program_id(1)
    for i in range(BLOCK_R):
        ir = pid_ir * BLOCK_R + i
        ir_mask = ir < R
        crow_start = tl.load(CROW_INDICES + n * stride_crow_n + ir *
            stride_crow_r, mask=ir_mask)
        crow_end = tl.load(CROW_INDICES + n * stride_crow_n + (ir + 1) *
            stride_crow_r, mask=ir_mask)
        row_mask = tl.arange(0, BLOCK_Z) + crow_start < crow_end
        row = tl.load(IN_VALUES + n * stride_in_n + (tl.arange(0, BLOCK_Z) +
            crow_start) * stride_in_z, mask=row_mask & ir_mask, other=-
            float('inf'))
        col_idx = tl.load(COL_INDICES + n * stride_col_n + (tl.arange(0,
            BLOCK_Z) + crow_start) * stride_col_z, mask=row_mask & ir_mask,
            other=0)
        head_idx = col_idx // T_SRC
        output = tl.zeros_like(row)
        for ih in range(H):
            head_mask = head_idx == ih
            row_per_head = tl.where(head_mask, row, -float('inf'))
            row_max = tl.max(row_per_head)
            row_minus_max = row_per_head - row_max
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator)
            softmax_result = numerator / denominator
            output += tl.where(head_mask, softmax_result, 0)
        tl.store(OUT_VALUES + n * stride_out_n + (tl.arange(0, BLOCK_Z) +
            crow_start) * stride_out_z, output, mask=row_mask & ir_mask)

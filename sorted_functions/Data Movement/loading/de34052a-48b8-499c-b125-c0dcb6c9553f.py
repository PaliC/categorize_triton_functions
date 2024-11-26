import triton
import triton.language as tl
import torch

@triton.jit
def mload2d(REG_ROWS: 'tl.constexpr', REG_COLS: 'tl.constexpr', i_base,
    i_start_row, i_start_col, i_rows, i_cols, stride_row, stride_col):
    off_rows = tl.arange(0, REG_ROWS) + i_start_row
    off_cols = tl.arange(0, REG_COLS) + i_start_col
    i_ptrs = i_base + off_rows[:, None] * stride_row + off_cols[None, :
        ] * stride_col
    row_overflow = i_start_row + REG_ROWS - i_rows
    col_overflow = i_start_col + REG_COLS - i_cols
    i_ptrs_mask = tl.full([REG_ROWS, REG_COLS], 1, dtype=tl.int1)
    if row_overflow > 0:
        i_ptrs_mask = i_ptrs_mask & (off_rows[:, None] < i_rows)
    if col_overflow > 0:
        i_ptrs_mask = i_ptrs_mask & (off_cols[None, :] < i_cols)
    return tl.load(i_ptrs, mask=i_ptrs_mask, other=0.0)

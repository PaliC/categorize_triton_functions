import triton
import triton.language as tl
import torch

@triton.jit
def _rms_kernel_bwd_dw(input_ptr, dout_ptr, dweight_ptr, rstdev_ptr, nrows,
    ncols, block_size_row: 'tl.constexpr', block_size_col: 'tl.constexpr'):
    row_index = tl.program_id(0)
    cols = row_index * block_size_col + tl.arange(0, block_size_col)
    dw = tl.zeros((block_size_row, block_size_col), dtype=tl.float32)
    unroll: 'tl.constexpr' = 4
    for outer in range(0, nrows, block_size_row * unroll):
        for inner in range(unroll):
            rows = outer + inner * block_size_row + tl.arange(0, block_size_row
                )
            mask = rows[:, None] < block_size_row & (cols[None, :] <
                block_size_col)
            offsets = rows[:, None] * block_size_col + cols[None, :]
            input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0)
            rstdev = tl.load(rstdev_ptr + rows, mask=rows < block_size_row,
                other=0.0)
            input_pred = input * rstdev[:, None]
            dw += dout * input_pred
    sum_dw = tl.sum(dw, axis=0)
    tl.store(dweight_ptr, sum_dw, mask=cols < block_size_col)

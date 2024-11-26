import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'bs_row': 256, 'bs_col': 256,
    'group_sz': 8}, num_warps=8), triton.Config({'bs_row': 128, 'bs_col': 
    128, 'group_sz': 8}, num_warps=8), triton.Config({'bs_row': 64,
    'bs_col': 64, 'group_sz': 8}, num_warps=8), triton.Config({'bs_row': 32,
    'bs_col': 32, 'group_sz': 8}, num_warps=8), triton.Config({'bs_row': 16,
    'bs_col': 16, 'group_sz': 8}, num_warps=8), triton.Config({'bs_row': 
    128, 'bs_col': 128, 'group_sz': 8}, num_warps=4), triton.Config({
    'bs_row': 64, 'bs_col': 64, 'group_sz': 8}, num_warps=4), triton.Config
    ({'bs_row': 32, 'bs_col': 32, 'group_sz': 8}, num_warps=4), triton.
    Config({'bs_row': 16, 'bs_col': 16, 'group_sz': 8}, num_warps=4),
    triton.Config({'bs_row': 256, 'bs_col': 256, 'group_sz': 4}, num_warps=
    8), triton.Config({'bs_row': 128, 'bs_col': 128, 'group_sz': 4},
    num_warps=8), triton.Config({'bs_row': 64, 'bs_col': 64, 'group_sz': 4},
    num_warps=8), triton.Config({'bs_row': 32, 'bs_col': 32, 'group_sz': 4},
    num_warps=8), triton.Config({'bs_row': 16, 'bs_col': 16, 'group_sz': 4},
    num_warps=8), triton.Config({'bs_row': 128, 'bs_col': 128, 'group_sz': 
    4}, num_warps=4), triton.Config({'bs_row': 64, 'bs_col': 64, 'group_sz':
    4}, num_warps=4), triton.Config({'bs_row': 32, 'bs_col': 32, 'group_sz':
    4}, num_warps=4), triton.Config({'bs_row': 16, 'bs_col': 16, 'group_sz':
    4}, num_warps=4)], key=['num_batches', 'num_rows', 'num_cols'])
@triton.jit
def add_kernel(input1_ptr, input2_ptr, input_batch_stride, input_row_stride,
    input_col_stride, num_batches, num_rows, num_cols, out_ptr, bs_row:
    'tl.constexpr', bs_col: 'tl.constexpr', group_sz: 'tl.constexpr'):
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)
    batch_offset = batch_idx * input_batch_stride
    row_offset = row_idx * bs_row + tl.arange(0, bs_row)
    col_offset = col_idx * bs_col + tl.arange(0, bs_col)
    data_offset = row_offset[:, None] * input_row_stride + col_offset[None, :
        ] * input_col_stride
    row_mask = row_offset < num_rows
    col_mask = col_offset < num_cols
    data_mask = row_mask[:, None] & col_mask[None, :]
    input1 = tl.load(input1_ptr + batch_offset + data_offset, data_mask)
    input2 = tl.load(input2_ptr + batch_offset + data_offset, data_mask)
    add = input1 + input2
    tl.store(out_ptr + batch_offset + data_offset, add, mask=data_mask)

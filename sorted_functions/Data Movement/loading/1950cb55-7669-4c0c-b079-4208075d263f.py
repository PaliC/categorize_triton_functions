import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_fwd_kernel(output_ptr, source_ptr, index_ptr,
    num_indices, num_cols, stride0, stride1, BLOCK_SIZE_INDEX:
    'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    rows = tl.load(index_ptr + indices, mask=indices < num_indices)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :
        ] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)
    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :
        ] * stride1
    tl.store(output_offsets, output, mask=mask)

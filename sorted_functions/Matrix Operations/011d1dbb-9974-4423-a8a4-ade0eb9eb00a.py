import triton
import triton.language as tl
import torch

@triton.jit
def scaled_index_add_fwd_kernel(input_ptr, index_ptr, source_ptr,
    scaling_ptr, alpha, num_inp_indices, num_src_indices, num_rows,
    num_cols, stride0, stride1, stride2, BLOCK_SIZE_INDEX: 'tl.constexpr',
    BLOCK_SIZE_ROW: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr',
    HAS_SCALING: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)
    rows = pid1 * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    cols = pid2 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    source_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    source_offsets = source_ptr + source_indices[:, None, None
        ] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :
        ] * stride2
    source_mask = (source_indices[:, None, None] < num_src_indices) & (rows
        [None, :, None] < num_rows) & (cols[None, None, :] < num_cols)
    source = tl.load(source_offsets, mask=source_mask)
    input_indices = tl.load(index_ptr + source_indices, mask=source_indices <
        num_src_indices)
    input_offsets = input_ptr + input_indices[:, None, None] * stride0 + rows[
        None, :, None] * stride1 + cols[None, None, :] * stride2
    x = tl.load(input_offsets, mask=source_mask)
    if HAS_SCALING:
        scaling = tl.load(scaling_ptr + cols[None, None, :] * stride2, mask
            =cols[None, None, :] < num_cols)
        tl.store(input_offsets, x + alpha * scaling * source, mask=source_mask)
    else:
        tl.store(input_offsets, x + alpha * source, mask=source_mask)

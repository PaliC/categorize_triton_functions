import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(grad_source_ptr, index_ptr, grad_output_ptr,
    num_rows, num_indices, num_cols, stride0, stride1, BLOCK_SIZE_INDEX:
    'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    grad_output_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0,
        BLOCK_SIZE_INDEX)
    grad_output_offsets = grad_output_ptr + grad_output_indices[:, None
        ] * stride0 + cols[None, :] * stride1
    grad_output_mask = (grad_output_indices[:, None] < num_indices) & (cols
        [None, :] < num_cols)
    grad_output = tl.load(grad_output_offsets, mask=grad_output_mask)
    grad_source_indices = tl.load(index_ptr + grad_output_indices, mask=
        grad_output_indices < num_indices)
    grad_source_offsets = grad_source_ptr + grad_source_indices[:, None
        ] * stride0 + cols[None, :] * stride1
    tl.store(grad_source_offsets, grad_output, mask=grad_output_mask)

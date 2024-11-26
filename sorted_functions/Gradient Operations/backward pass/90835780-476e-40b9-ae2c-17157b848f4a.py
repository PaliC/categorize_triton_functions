import triton
import triton.language as tl
import torch

@triton.jit
def scaled_index_add_bwd_kernel(grad_output_ptr, grad_source_ptr,
    grad_scaling_ptr, source_ptr, scaling_ptr, index_ptr, alpha,
    num_inp_indices, num_src_indices, num_rows, num_cols, stride0, stride1,
    stride2, BLOCK_SIZE_INDEX: 'tl.constexpr', BLOCK_SIZE_ROW:
    'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr', HAS_SCALING: 'tl.constexpr'
    ):
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
    grad_output_indices = tl.load(index_ptr + source_indices, mask=
        source_indices < num_src_indices)
    grad_output_offsets = (grad_output_ptr + grad_output_indices * stride0 +
        rows[None, :, None] * stride1 + cols[None, None, :] * stride2)
    grad_output = tl.load(grad_output_offsets, mask=source_mask)
    grad_source_offsets = grad_source_ptr + source_indices[:, None, None
        ] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :
        ] * stride2
    if HAS_SCALING:
        scaling = tl.load(scaling_ptr + cols[None, None, :] * stride2, mask
            =cols[None, None, :] < num_cols)
        tl.store(grad_source_offsets, alpha * grad_output * scaling, mask=
            source_mask)
        grad_scaling_offsets = grad_scaling_ptr + source_indices[:, None, None
            ] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :
            ] * stride2
        tl.store(grad_scaling_offsets, alpha * grad_output * source, mask=
            source_mask)
    else:
        tl.store(grad_source_offsets, alpha * grad_output, mask=source_mask)

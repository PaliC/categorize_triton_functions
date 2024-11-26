import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64,
    'group_sz': 8}, num_stages=3, num_warps=8), triton.Config({'bsy': 64,
    'bsx': 256, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 8},
    num_stages=4, num_warps=4), triton.Config({'bsy': 64, 'bsx': 128, 'bsk':
    32, 'group_sz': 8}, num_stages=4, num_warps=4), triton.Config({'bsy': 
    64, 'bsx': 32, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
    triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 8},
    num_stages=5, num_warps=2), triton.Config({'bsy': 128, 'bsx': 256,
    'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8), triton.Config({
    'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=3,
    num_warps=8), triton.Config({'bsy': 256, 'bsx': 64, 'bsk': 128,
    'group_sz': 8}, num_stages=4, num_warps=4), triton.Config({'bsy': 64,
    'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 8},
    num_stages=4, num_warps=4), triton.Config({'bsy': 64, 'bsx': 128, 'bsk':
    64, 'group_sz': 8}, num_stages=4, num_warps=4), triton.Config({'bsy': 
    128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 4},
    num_stages=3, num_warps=8), triton.Config({'bsy': 64, 'bsx': 256, 'bsk':
    32, 'group_sz': 4}, num_stages=4, num_warps=4), triton.Config({'bsy': 
    128, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
    triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 4},
    num_stages=4, num_warps=4), triton.Config({'bsy': 128, 'bsx': 32, 'bsk':
    32, 'group_sz': 4}, num_stages=4, num_warps=4), triton.Config({'bsy': 
    32, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=5, num_warps=2),
    triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 4},
    num_stages=3, num_warps=8), triton.Config({'bsy': 256, 'bsx': 128,
    'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8), triton.Config({
    'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=4,
    num_warps=4), triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128,
    'group_sz': 4}, num_stages=4, num_warps=4), triton.Config({'bsy': 64,
    'bsx': 128, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4},
    num_stages=4, num_warps=4)], key=['batch_size', 'seq_len', 'dim',
    'dim_out'])
@triton.jit
def matmul_kernel(A_ptr, B_ptr, O_ptr, A_stride_batch, A_stride_height,
    A_stride_width, B_stride_batch, B_stride_height, B_stride_width,
    O_stride_batch, O_stride_height, O_stride_width, batch_size, seq_len,
    dim, dim_out, bsx: 'tl.constexpr', bsy: 'tl.constexpr', bsk:
    'tl.constexpr', group_sz: 'tl.constexpr', apply_scaling: 'tl.constexpr',
    scale_factor: 'tl.constexpr'):
    """
    Matrix multiplication by loading rows of A
    and columns of B to calculate a block of O.

    This can be further improved by implementing tiling, however
    I am yet to figure out how to use L2 cache in Triton.
    """
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)
    num_row_programs = tl.num_programs(1)
    num_col_programs = tl.num_programs(2)
    row_idxnew, col_idxnew = tl.swizzle2d(row_idx, col_idx,
        num_row_programs, num_col_programs, group_sz)
    a_offset_batch = batch_idx * A_stride_batch
    b_offset_batch = batch_idx * B_stride_batch
    output = tl.zeros((bsy, bsx), dtype=tl.float32)
    for offset in range(0, dim, bsk):
        offset_k = offset + tl.arange(0, bsk)
        offset_a = row_idxnew * bsy + tl.arange(0, bsy)
        mask_a = (offset_a[:, None] < seq_len) & (offset_k[None, :] < dim)
        offset_a = offset_a[:, None] * A_stride_height + offset_k[None, :
            ] * A_stride_width
        a = tl.load(A_ptr + a_offset_batch + offset_a, mask_a)
        offset_b = col_idxnew * bsx + tl.arange(0, bsx)
        mask_b = (offset_k[:, None] < dim) & (offset_b[None, :] < dim_out)
        offset_b = offset_k[:, None] * B_stride_height + offset_b[None, :
            ] * B_stride_width
        b = tl.load(B_ptr + b_offset_batch + offset_b, mask_b)
        output = tl.dot(a, b, output, allow_tf32=True)
    offset_out_batch = batch_idx * O_stride_batch
    offset_or = row_idxnew * bsy + tl.arange(0, bsy)
    offset_oc = col_idxnew * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None] * O_stride_height + offset_oc[None, :
        ] * O_stride_width
    mask_o = (offset_or[:, None] < seq_len) & (offset_oc[None, :] < dim_out)
    if apply_scaling:
        output = scale_factor * output
    tl.store(O_ptr + offset_out_batch + offset_o, output, mask_o)

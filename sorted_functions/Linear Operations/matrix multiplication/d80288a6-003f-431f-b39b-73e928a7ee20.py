import triton
import triton.language as tl
import torch

@triton.jit
def blora_fwd_kernel(x_ptr, x_stride_bsk, x_stride_m, x_stride_h, o_ptr,
    o_stride_bsk, o_stride_m, o_stride_hout, blA_ptr, blA_stride_bsk,
    blA_stride_h, blA_stride_r, blB_ptr, blB_stride_bsk, blB_stride_r,
    blB_stride_hout, h: 'tl.constexpr', hout: 'tl.constexpr', m:
    'tl.constexpr', r: 'tl.constexpr', block_size_h: 'tl.constexpr',
    block_size_hout: 'tl.constexpr', block_size_m: 'tl.constexpr',
    block_size_r: 'tl.constexpr'):
    block_idx_bsk = tl.program_id(0)
    block_idx_hout = tl.program_id(1)
    offsets_h = tl.arange(0, block_size_h)
    offsets_hout = block_idx_hout * block_size_hout + tl.arange(0,
        block_size_hout)
    offsets_m = tl.arange(0, block_size_m)
    offsets_r = tl.arange(0, block_size_r)
    block_mask_m_col = offsets_m[:, None] < m
    block_mask_r_row = offsets_r[None, :] < r
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_hout_row = offsets_hout[None, :] < hout
    x_ptrs = x_ptr + block_idx_bsk * x_stride_bsk + (offsets_m[:, None] *
        x_stride_m + offsets_h[None, :] * x_stride_h)
    o_ptrs = o_ptr + block_idx_bsk * o_stride_bsk + (offsets_m[:, None] *
        o_stride_m + offsets_hout[None, :] * o_stride_hout)
    blA_ptrs = blA_ptr + block_idx_bsk * blA_stride_bsk + (offsets_h[:,
        None] * blA_stride_h + offsets_r[None, :] * blA_stride_r)
    blB_ptrs = blB_ptr + block_idx_bsk * blB_stride_bsk + (offsets_r[:,
        None] * blB_stride_r + offsets_hout[None, :] * blB_stride_hout)
    olA = tl.zeros((block_size_m, block_size_r), dtype=tl.float32)
    for block_idx_h in range(0, tl.cdiv(h, block_size_h)):
        block_offs_h = block_idx_h * block_size_h + offsets_h
        block_mask_h_row = block_offs_h[None, :] < h
        block_mask_h_col = block_offs_h[:, None] < h
        x = tl.load(x_ptrs, mask=block_mask_m_col & block_mask_h_row, other=0.0
            )
        blA = tl.load(blA_ptrs, mask=block_mask_h_col & block_mask_r_row,
            other=0.0)
        olA += tl.dot(x, blA)
        x_ptrs += block_size_h * x_stride_h
        blA_ptrs += block_size_h * blA_stride_h
    blB = tl.load(blB_ptrs, mask=block_mask_r_col & block_mask_hout_row,
        other=0.0)
    olB = tl.dot(olA, blB)
    tl.store(o_ptrs, olB, mask=block_mask_m_col & block_mask_hout_row)

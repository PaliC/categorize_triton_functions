import triton
import triton.language as tl
import torch

@triton.jit
def blora_bp_kernel(dr_ptr, dr_stride_bsk, dr_stride_m, dr_stride_hout,
    dx_ptr, dx_stride_bsk, dx_stride_m, dx_stride_h, blA_ptr,
    blA_stride_bsk, blA_stride_h, blA_stride_r, blB_ptr, blB_stride_bsk,
    blB_stride_r, blB_stride_hout, h: 'tl.constexpr', hout: 'tl.constexpr',
    m: 'tl.constexpr', r: 'tl.constexpr', block_size_h: 'tl.constexpr',
    block_size_hout: 'tl.constexpr', block_size_m: 'tl.constexpr',
    block_size_r: 'tl.constexpr'):
    block_idx_bsk = tl.program_id(0)
    block_idx_h = tl.program_id(1)
    offsets_h = block_idx_h * block_size_h + tl.arange(0, block_size_h)
    offsets_hout = tl.arange(0, block_size_hout)
    offsets_m = tl.arange(0, block_size_m)
    offsets_r = tl.arange(0, block_size_r)
    block_mask_m_col = offsets_m[:, None] < m
    block_mask_r_row = offsets_r[None, :] < r
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_h_row = offsets_h[None, :] < h
    block_mask_h_col = offsets_h[:, None] < h
    dx_ptrs = dx_ptr + block_idx_bsk * dx_stride_bsk + (offsets_m[:, None] *
        dx_stride_m + offsets_h[None, :] * dx_stride_h)
    dr_ptrs = dr_ptr + block_idx_bsk * dr_stride_bsk + (offsets_m[:, None] *
        dr_stride_m + offsets_hout[None, :] * dr_stride_hout)
    blA_ptrs = blA_ptr + block_idx_bsk * blA_stride_bsk + (offsets_h[:,
        None] * blA_stride_h + offsets_r[None, :] * blA_stride_r)
    blB_ptrs = blB_ptr + block_idx_bsk * blB_stride_bsk + (offsets_r[:,
        None] * blB_stride_r + offsets_hout[None, :] * blB_stride_hout)
    olB = tl.zeros((block_size_r, block_size_m), dtype=tl.float32)
    for block_idx_hout in range(0, tl.cdiv(hout, block_size_hout)):
        block_offs_hout = block_idx_hout * block_size_hout + offsets_hout
        block_mask_hout_row = block_offs_hout[None, :] < hout
        block_mask_hout_col = block_offs_hout[:, None] < hout
        dr = tl.load(dr_ptrs, mask=block_mask_m_col & block_mask_hout_row,
            other=0.0)
        blB = tl.load(blB_ptrs, mask=block_mask_r_col & block_mask_hout_row,
            other=0.0)
        olB += tl.dot(blB, dr.T)
        dr_ptrs += block_size_hout * dr_stride_hout
        blB_ptrs += block_size_hout * blB_stride_hout
    blA = tl.load(blA_ptrs, mask=block_mask_h_col & block_mask_r_row, other=0.0
        )
    olA = tl.dot(blA, olB)
    tl.store(dx_ptrs, olA.T, mask=block_mask_m_col & block_mask_h_row)

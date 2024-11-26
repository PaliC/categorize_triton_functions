import triton
import triton.language as tl
import torch

@triton.jit
def blora_bp_kernel_without_mask(dr_ptr, dr_stride_bs, dx_ptr,
    dx_stride_bsk, dx_stride_hm, lA_ptr, lA_stride_l, lA_stride_hm, lB_ptr,
    lB_stride_l, lB_stride_r, sel_ptr, k: 'tl.constexpr', m: 'tl.constexpr',
    r: 'tl.constexpr', rm: 'tl.constexpr', hm: 'tl.constexpr', hout:
    'tl.constexpr', block_size_hout: 'tl.constexpr', block_size_hm:
    'tl.constexpr', block_size_r: 'tl.constexpr'):
    block_idx_bsk = tl.program_id(0)
    block_idx_bs = block_idx_bsk // k
    block_idx_hm = tl.program_id(1)
    offsets_hout = tl.arange(0, block_size_hout)
    offsets_hm = block_idx_hm * block_size_hm + tl.arange(0, block_size_hm)
    offsets_r = tl.arange(0, block_size_r)
    offsets_m = tl.arange(0, m)
    offsets_rm = tl.arange(0, rm)
    block_mask_hout = offsets_hout < hout
    block_mask_hout_row = offsets_hout[None, :] < hout
    block_mask_hm_col = offsets_hm[:, None] < hm
    block_mask_rm_row = offsets_rm[None, :] < rm
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_m_row = offsets_r[None, :] < m
    sel_ptr += block_idx_bsk
    sel_idx = tl.load(sel_ptr)
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + (offsets_hm[:, None] *
        lA_stride_hm + offsets_rm[None, :])
    dr_block_ptrs = dr_ptr + block_idx_bs * dr_stride_bs + ([[0]] +
        offsets_hout[None, :])
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + (offsets_hm[:, None] *
        lA_stride_hm + offsets_rm)
    lB_block_ptrs = lB_ptr + sel_idx * lB_stride_l + (offsets_r[:, None] *
        lB_stride_r + offsets_hout[None, :])
    dx_block_ptrs = dx_ptr + block_idx_bsk * dx_stride_bsk + offsets_hm[:, None
        ] * dx_stride_hm + offsets_m[None, :]
    compute_olB_dtype = tl.float16
    olB = tl.zeros((block_size_r, 1))
    for _ in range(tl.cdiv(hout, block_size_hout)):
        block_mask_hout_col = offsets_hout[:, None] < hout
        block_mask_hout_row = offsets_hout[None, :] < hout
        dr = tl.load(dr_block_ptrs, mask=block_mask_hout_row, other=0.0)
        lB = tl.load(lB_block_ptrs, mask=block_mask_hout_row, others=0.0)
        olB += tl.dot(lB, dr.T)
        offsets_hout += block_size_hout
        dr_block_ptrs += block_size_hout
        lB_block_ptrs += block_size_hout
    olB_r = torch.zeros(r * m, m, dtype=compute_olB_dtype, device=olB.device)
    for i in range(m):
        olB_r[i * r:(i + 1) * r, i] = olB
    compute_olA_dtype = tl.float16
    lA = tl.load(lA_block_ptrs, mask=block_mask_hm_col & block_mask_rm_row,
        other=0.0)
    olA = tl.dot(lA, olB)
    tl.store(dx_block_ptrs, olA, mask=block_mask_hm_col & block_mask_m_row)

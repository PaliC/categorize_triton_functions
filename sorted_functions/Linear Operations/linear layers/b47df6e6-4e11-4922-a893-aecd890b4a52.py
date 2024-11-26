import triton
import triton.language as tl
import torch

@triton.jit
def blora_bp_kernel_with_loraB_mask(dr_ptr, dr_stride_bs, dr_stride_n,
    dx_ptr, dx_stride_bsk, lA_ptr, lA_stride_l, lA_stride_h, lB_ptr,
    lB_stride_l, lB_stride_rn, lB_mask1_ptr, lB_mask1_stride_rn,
    lB_mask2_ptr, lB_mask2_stride_r, sel_ptr, k: 'tl.constexpr', n:
    'tl.constexpr', r: 'tl.constexpr', rn: 'tl.constexpr', hn:
    'tl.constexpr', h: 'tl.constexpr', block_size_r: 'tl.constexpr',
    block_size_hn: 'tl.constexpr', block_size_h: 'tl.constexpr'):
    """ 
    dr shape = (bs, n, hout//n)
    lora_B_weights shape = (loras, r, hout) -> (loras, rn, hout//n)
    lora_A_weights shape = (loras, h, r)
    """
    block_idx_bsk = tl.program_id(0)
    block_idx_bs = block_idx_bsk // k
    block_idx_h = tl.program_id(1)
    offsets_h = block_idx_h * block_size_h + tl.arange(0, block_size_h)
    offsets_hn = tl.arange(0, block_size_hn)
    offsets_r = tl.arange(0, block_size_r)
    offsets_n = tl.arange(0, n)
    offsets_rn = tl.arange(0, rn)
    block_mask_h = offsets_h < h
    block_mask_h_col = offsets_h[:, None] < h
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_r_row = offsets_r[None, :] < r
    sel_ptr += block_idx_bsk
    sel_idx = tl.load(sel_ptr)
    dr_block_ptrs = dr_ptr + block_idx_bs * dr_stride_bs + (offsets_n[:,
        None] * dr_stride_n + offsets_hn[None, :])
    lB_block_ptrs = lB_ptr + sel_idx * lB_stride_l + (offsets_rn[:, None] *
        lB_stride_rn + offsets_hn[None, :])
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + (offsets_h[:, None] *
        lA_stride_h + offsets_r[None, :])
    lB_mask1_ptrs = lB_mask1_ptr + (offsets_rn[:, None] *
        lB_mask1_stride_rn + offsets_n[None, :])
    lB_mask2_ptrs = lB_mask2_ptr + (offsets_r[:, None] * lB_mask2_stride_r +
        offsets_rn[None, :])
    dx_block_ptrs = dx_ptr + block_idx_bsk * dx_stride_bsk + offsets_h
    compute_olB_dtype = tl.float16
    olB = tl.zeros((rn, n), dtype=compute_olB_dtype)
    for block_idx_hn in range(tl.cdiv(hn, block_size_hn)):
        block_mask_hn_col = offsets_hn[:, None] < hn
        block_mask_hn_row = offsets_hn[None, :] < hn
        dr = tl.load(dr_block_ptrs, mask=block_mask_hn_row, other=0.0)
        lB = tl.load(lB_block_ptrs, mask=block_mask_hn_row, other=0.0)
        olB += tl.dot(lB, dr.T)
        offsets_hn += block_size_hn
        dr_block_ptrs += block_size_hn
        lB_block_ptrs += block_size_hn
    lB_mask1 = tl.load(lB_mask1_ptrs)
    lB_mask2 = tl.load(lB_mask2_ptrs, mask=block_mask_r_col, other=0.0)
    olB = tl.dot(lB_mask2, olB * lB_mask1)
    compute_olA_dtype = tl.float16
    lA = tl.load(lA_block_ptrs, mask=block_mask_h_col & block_mask_r_row,
        other=0.0)
    olA = tl.sum(tl.dot(lA, olB), axis=1)
    tl.store(dx_block_ptrs, olA, mask=block_mask_h)

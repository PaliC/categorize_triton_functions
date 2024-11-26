import triton
import triton.language as tl
import torch

@triton.jit
def blora_fwd_kernel_with_loraA_mask(x_ptr, x_stride_bs, x_stride_m, o_ptr,
    o_stride_bsk, lA_ptr, lA_stride_l, lA_stride_hm, lB_ptr, lB_stride_l,
    lB_stride_r, lA_mask1_ptr, lA_mask1_stride_m, lA_mask2_ptr,
    lA_mask2_stride_rm, sel_ptr, k: 'tl.constexpr', m: 'tl.constexpr', r:
    'tl.constexpr', rm: 'tl.constexpr', hm: 'tl.constexpr', hout:
    'tl.constexpr', block_size_hout: 'tl.constexpr', block_size_hm:
    'tl.constexpr', block_size_r: 'tl.constexpr'):
    block_idx_bsk = tl.program_id(0)
    block_idx_bs = block_idx_bsk // k
    block_idx_hout = tl.program_id(1)
    offsets_hout = block_idx_hout * block_size_hout + tl.arange(0,
        block_size_hout)
    offsets_hm = tl.arange(0, block_size_hm)
    offsets_r = tl.arange(0, block_size_r)
    offsets_m = tl.arange(0, m)
    offsets_rm = tl.arange(0, rm)
    block_mask_hout = offsets_hout < hout
    block_mask_hout_row = offsets_hout[None, :] < hout
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_r_row = offsets_r[None, :] < r
    sel_ptr += block_idx_bsk
    sel_idx = tl.load(sel_ptr)
    x_block_ptrs = x_ptr + block_idx_bs * x_stride_bs + (offsets_m[:, None] *
        x_stride_m + offsets_hm[None, :])
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + (offsets_hm[:, None] *
        lA_stride_hm + offsets_rm[None, :])
    lB_block_ptrs = lB_ptr + sel_idx * lB_stride_l + (offsets_r[:, None] *
        lB_stride_r + offsets_hout[None, :])
    lA_mask1_ptrs = lA_mask1_ptr + (offsets_m[:, None] * lA_mask1_stride_m +
        offsets_rm[None, :])
    lA_mask2_ptrs = lA_mask2_ptr + (offsets_rm[:, None] *
        lA_mask2_stride_rm + offsets_r[None, :])
    o_block_ptrs = o_ptr + block_idx_bsk * o_stride_bsk + offsets_hout
    compute_olA_dtype = tl.float16
    olA = tl.zeros((m, rm), dtype=compute_olA_dtype)
    for block_idx_hm in range(tl.cdiv(hm, block_size_hm)):
        block_mask_hm_col = offsets_hm[:, None] < hm
        block_mask_hm_row = offsets_hm[None, :] < hm
        x = tl.load(x_block_ptrs, mask=block_mask_hm_row, other=0.0)
        lA = tl.load(lA_block_ptrs, mask=block_mask_hm_col, other=0.0)
        olA += tl.dot(x, lA)
        offsets_hm += block_size_hm
        x_block_ptrs += block_size_hm
        lA_block_ptrs += block_size_hm * lA_stride_hm
    lA_mask1 = tl.load(lA_mask1_ptrs)
    lA_mask2 = tl.load(lA_mask2_ptrs, mask=block_mask_r_row, other=0.0)
    olA = tl.dot(olA * lA_mask1, lA_mask2)
    compute_olB_dtype = tl.float16
    lB = tl.load(lB_block_ptrs, mask=block_mask_r_col & block_mask_hout_row,
        other=0.0)
    olB = tl.sum(tl.dot(olA, lB), axis=0)
    tl.store(o_block_ptrs, olB, mask=block_mask_hout)

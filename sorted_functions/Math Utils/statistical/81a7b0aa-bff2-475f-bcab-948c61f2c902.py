import triton
import triton.language as tl
import torch

@triton.jit
def _kernel_bwd_merge_continuous(alpha_c, alpha_d, tmp_merge,
    tmp_merge_normalized, tmp_merge_grad, stride_alpha_c1, stride_alpha_c2,
    stride_alpha_c3, stride_alpha_d1, stride_alpha_d2, stride_alpha_d3,
    stride_alpha_d4, stride_alpha_d5, stride_tmp_merge1, stride_tmp_merge2,
    r1, r2, r3, r4, b, n, w, L, BLOCK_R1: 'tl.constexpr', BLOCK_R2:
    'tl.constexpr'):
    b_idx = tl.program_id(0)
    start = tl.program_id(1)
    end = start + w
    if b_idx >= b:
        return
    offset_r = tl.arange(0, BLOCK_R1)
    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (
        start + 1) * stride_alpha_c3 + offset_r
    l_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + (start + 1
        ) * stride_alpha_c2 + start * stride_alpha_c3 + offset_r
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start + 1
        ) * stride_alpha_c2 + end * stride_alpha_c3 + r1 + offset_r
    r_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + end * stride_alpha_c2 + (
        start + 1) * stride_alpha_c3 + r1 + offset_r
    mask = tl.arange(0, BLOCK_R1) < r1
    parent_score = tl.load(tmp_merge + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + tl.arange(0, BLOCK_R1), mask=mask, other=0)
    do = tl.load(tmp_merge_grad + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + tl.arange(0, BLOCK_R1), mask=mask, other=0)
    do *= tl.load(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + tl.arange(0, BLOCK_R1), mask=mask, other=0)
    for _ in range(0, w - 1):
        left_score = tl.load(l_ptr, mask=mask, other=0)
        right_score = tl.load(r_ptr, mask=mask, other=0)
        new_grad = tl.exp(left_score + right_score - parent_score) * do
        tl.atomic_add(l_bwd_ptr, new_grad, mask=mask)
        tl.atomic_add(r_bwd_ptr, new_grad, mask=mask)
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2
        l_bwd_ptr += stride_alpha_c2
        r_bwd_ptr += stride_alpha_c3
    mask2 = tl.arange(0, BLOCK_R2) < r2
    parent_score = tl.load(tmp_merge + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), mask=mask2, other=0)
    do = tl.load(tmp_merge_grad + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), mask=mask2, other=0)
    do *= tl.load(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), mask=mask2, other=0)
    for gap_start in range(start + 1, end - 1):
        for gap_end in range(gap_start + 1, end):
            ptr_c = (alpha_c + b_idx * stride_alpha_c1 + gap_start *
                stride_alpha_c2 + gap_end * stride_alpha_c3 + 2 * r1 + tl.
                arange(0, BLOCK_R2))
            ptr_d = (alpha_d + b_idx * stride_alpha_d1 + start *
                stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end *
                stride_alpha_d4 + end * stride_alpha_d5 + tl.arange(0,
                BLOCK_R2))
            cont = tl.load(ptr_c, mask=mask2, other=0)
            disco = tl.load(ptr_d, mask=mask2, other=0)
            new_grad = tl.exp(cont + disco - parent_score) * do
            ptr_bwd_c = (alpha_c + b_idx * stride_alpha_c1 + gap_end *
                stride_alpha_c2 + gap_start * stride_alpha_c3 + 2 * r1 + tl
                .arange(0, BLOCK_R2))
            ptr_bwd_d = (alpha_d + b_idx * stride_alpha_d1 + gap_start *
                stride_alpha_d2 + start * stride_alpha_d3 + gap_end *
                stride_alpha_d4 + end * stride_alpha_d5 + tl.arange(0,
                BLOCK_R2))
            tl.atomic_add(ptr_bwd_c, new_grad, mask=mask2)
            tl.atomic_add(ptr_bwd_d, new_grad, mask=mask2)

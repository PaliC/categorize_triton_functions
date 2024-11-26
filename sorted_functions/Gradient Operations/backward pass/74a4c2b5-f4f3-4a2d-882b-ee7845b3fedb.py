import triton
import triton.language as tl
import torch

@triton.jit
def _kernel_bwd_merge_discontinuous_v2(alpha_c, alpha_d, tmp_merge,
    tmp_merge_normalized, tmp_merge_grad, w, batch, L, stride_alpha_c1,
    stride_alpha_c2, stride_alpha_c3, stride_alpha_d1, stride_alpha_d2,
    stride_alpha_d3, stride_alpha_d4, stride_alpha_d5, stride_tmp_merge1,
    stride_tmp_merge2, stride_tmp_merge3, r1, r2, r3, r4, BLOCK_R3:
    'tl.constexpr', BLOCK_R4: 'tl.constexpr'):
    b_idx = tl.program_id(0)
    if b_idx >= batch:
        return
    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0
    while tid >= L - w - start:
        tid -= L - w - start
        start += 1
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)
    end = gap_end + (w - span_length_left)
    ptr = b_idx * stride_tmp_merge1 + tl.program_id(1
        ) * stride_tmp_merge2 + tl.program_id(2
        ) * stride_tmp_merge3 + tl.arange(0, BLOCK_R4)
    alpha_c_ptr = (alpha_c + b_idx * stride_alpha_c1 + 2 * r1 + r2 + 2 * r3 +
        tl.arange(0, BLOCK_R4))
    alpha_d_ptr = alpha_d + b_idx * stride_alpha_d1 + r2 + tl.arange(0,
        BLOCK_R4)
    mask = tl.arange(0, BLOCK_R4) < r4
    parent_score = tl.load(tmp_merge + ptr, mask=mask, other=0)
    do = tl.load(tmp_merge_normalized + ptr, mask=mask, other=0) * tl.load(
        tmp_merge_grad + ptr, mask=mask, other=0)
    for split in range(start + 1, gap_start):
        c_ptr = alpha_c_ptr + start * stride_alpha_c2 + split * stride_alpha_c3
        d_ptr = (alpha_d_ptr + split * stride_alpha_d2 + gap_start *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
            )
        child_c = tl.load(c_ptr, mask=mask, other=0)
        child_d = tl.load(d_ptr, mask=mask, other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do
        c_bwd_ptr = (alpha_c_ptr + split * stride_alpha_c2 + start *
            stride_alpha_c3)
        d_bwd_ptr = (alpha_d_ptr + gap_start * stride_alpha_d2 + split *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
            )
        tl.atomic_add(c_bwd_ptr, new_grad)
        tl.atomic_add(d_bwd_ptr, new_grad)
        c_ptr = (alpha_c_ptr + split * stride_alpha_c2 + gap_start *
            stride_alpha_c3 + r4)
        d_ptr = (alpha_d_ptr + start * stride_alpha_d2 + split *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
            )
        child_c = tl.load(c_ptr, mask=mask, other=0)
        child_d = tl.load(d_ptr, mask=mask, other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do
        c_bwd_ptr = (alpha_c_ptr + gap_start * stride_alpha_c2 + split *
            stride_alpha_c3 + r4)
        d_bwd_ptr = (alpha_d_ptr + split * stride_alpha_d2 + start *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
            )
        tl.atomic_add(c_bwd_ptr, new_grad)
        tl.atomic_add(d_bwd_ptr, new_grad)
    for split in range(gap_end + 1, end):
        c_ptr = (alpha_c_ptr + gap_end * stride_alpha_c2 + split *
            stride_alpha_c3 + 2 * r4)
        d_ptr = (alpha_d_ptr + start * stride_alpha_d2 + gap_start *
            stride_alpha_d3 + split * stride_alpha_d4 + end * stride_alpha_d5)
        child_c = tl.load(c_ptr, mask=mask, other=0)
        child_d = tl.load(d_ptr, mask=mask, other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do
        c_bwd_ptr = (alpha_c_ptr + split * stride_alpha_c2 + gap_end *
            stride_alpha_c3 + 2 * r4)
        d_bwd_ptr = (alpha_d_ptr + gap_start * stride_alpha_d2 + start *
            stride_alpha_d3 + split * stride_alpha_d4 + end * stride_alpha_d5)
        tl.atomic_add(c_bwd_ptr, new_grad)
        tl.atomic_add(d_bwd_ptr, new_grad)
        c_ptr = (alpha_c_ptr + split * stride_alpha_c2 + end *
            stride_alpha_c3 + 3 * r4)
        d_ptr = (alpha_d_ptr + start * stride_alpha_d2 + gap_start *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + split *
            stride_alpha_d5)
        child_c = tl.load(c_ptr, mask=mask, other=0)
        child_d = tl.load(d_ptr, mask=mask, other=0)
        new_grad = tl.exp(child_c + child_d - parent_score) * do
        c_bwd_ptr = (alpha_c_ptr + end * stride_alpha_c2 + split *
            stride_alpha_c3 + 3 * r4)
        d_bwd_ptr = (alpha_d_ptr + gap_start * stride_alpha_d2 + start *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + split *
            stride_alpha_d5)
        tl.atomic_add(c_bwd_ptr, new_grad)
        tl.atomic_add(d_bwd_ptr, new_grad)

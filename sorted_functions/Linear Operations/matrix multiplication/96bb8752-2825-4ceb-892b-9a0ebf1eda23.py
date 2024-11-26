import triton
import triton.language as tl
import torch

@triton.jit
def _kernel_inside_merge_discontinuous_v1(alpha_c, tmp_merge,
    tmp_merge_normalizer, w, batch, L, stride_alpha_c1, stride_alpha_c2,
    stride_alpha_c3, stride_tmp_merge1, stride_tmp_merge2,
    stride_tmp_merge3, stride_normalizer1, stride_normalizer2, r1, r2, r3,
    r4, BLOCK_R3: 'tl.constexpr'):
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
    l_ptr = (alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + 
        gap_start * stride_alpha_c3 + 2 * r1 + r2 + tl.arange(0, BLOCK_R3))
    r_ptr = (alpha_c + b_idx * stride_alpha_c1 + gap_end * stride_alpha_c2 +
        end * stride_alpha_c3 + 2 * r1 + r2 + r3 + tl.arange(0, BLOCK_R3))
    mask = tl.arange(0, BLOCK_R3) < r3
    child_l = tl.load(l_ptr, mask=mask, other=-1000000000.0)
    child_r = tl.load(r_ptr, mask=mask, other=-1000000000.0)
    acc1 = child_l + child_r
    acc_max = tl.max(acc1, 0)
    tl.store(tmp_merge_normalizer + b_idx * stride_normalizer1 + tl.
        program_id(1) * stride_normalizer2 + tl.program_id(2), acc_max)
    acc = tl.exp(acc1 - acc_max)
    tl.store(tmp_merge + b_idx * stride_tmp_merge1 + tl.program_id(1) *
        stride_tmp_merge2 + tl.program_id(2) * stride_tmp_merge3 + tl.
        arange(0, BLOCK_R3), acc, mask=mask)

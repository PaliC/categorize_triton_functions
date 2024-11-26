import triton
import triton.language as tl
import torch

@triton.jit
def _save_into_alpha_d(alpha_d, x, stride_alpha_d1, stride_alpha_d2,
    stride_alpha_d3, stride_alpha_d4, stride_alpha_d5, stride_merge1,
    stride_merge2, stride_merge3, B, L, w, r, BLOCK_RD: 'tl.constexpr'):
    b_idx = tl.program_id(0)
    if b_idx >= B:
        return
    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0
    mask = tl.arange(0, BLOCK_RD) < r
    to_save = tl.load(x + b_idx * stride_merge1 + tl.program_id(1) *
        stride_merge2 + tid * stride_merge3 + tl.arange(0, BLOCK_RD), mask=
        mask, other=0)
    while tid >= L - w - start:
        tid -= L - w - start
        start += 1
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)
    end = gap_end + (w - span_length_left)
    tl.store(alpha_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 + 
        gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end *
        stride_alpha_d5 + tl.arange(0, BLOCK_RD), to_save, mask=mask)

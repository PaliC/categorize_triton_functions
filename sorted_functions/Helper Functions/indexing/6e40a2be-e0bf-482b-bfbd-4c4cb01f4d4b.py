import triton
import triton.language as tl
import torch

@triton.jit
def _kernel_argmax_merge_discontinuous(alpha_c, alpha_d, marginal_d, w,
    batch, L, stride_alpha_c1, stride_alpha_c2, stride_alpha_c3,
    stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4,
    stride_alpha_d5):
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
    alpha_c_ptr = alpha_c + b_idx * stride_alpha_c1
    alpha_d_ptr = alpha_d + b_idx * stride_alpha_d1
    max_score = tl.load(alpha_c_ptr + start * stride_alpha_c2 + gap_start
        ) + tl.load(alpha_c_ptr + gap_end * stride_alpha_c2 + end)
    max_idx = tl.zeros((1,), dtype=tl.float32) - 1
    for split in range(start + 1, gap_start):
        c_ptr = alpha_c_ptr + start * stride_alpha_c2 + split * stride_alpha_c3
        d_ptr = (alpha_d_ptr + split * stride_alpha_d2 + gap_start *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
            )
        score = tl.load(c_ptr) + tl.load(d_ptr)
        max_idx = update_position(max_score, score, max_idx, tl.zeros((1,),
            dtype=tl.float32) + split)
        max_score = tl.maximum(score, max_score)
        c_ptr = (alpha_c_ptr + split * stride_alpha_c2 + gap_start *
            stride_alpha_c3)
        d_ptr = (alpha_d_ptr + start * stride_alpha_d2 + split *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + end * stride_alpha_d5
            )
        score = tl.load(c_ptr) + tl.load(d_ptr)
        max_idx = update_position(max_score, score, max_idx, tl.zeros((1,),
            dtype=tl.float32) + split + L + 1)
        max_score = tl.maximum(score, max_score)
    for split in range(gap_end + 1, end):
        c_ptr = (alpha_c_ptr + gap_end * stride_alpha_c2 + split *
            stride_alpha_c3)
        d_ptr = (alpha_d_ptr + start * stride_alpha_d2 + gap_start *
            stride_alpha_d3 + split * stride_alpha_d4 + end * stride_alpha_d5)
        score = tl.load(c_ptr) + tl.load(d_ptr)
        max_idx = update_position(max_score, score, max_idx, tl.zeros((1,),
            dtype=tl.float32) + split + 2 * (L + 1))
        max_score = tl.maximum(score, max_score)
        c_ptr = alpha_c_ptr + split * stride_alpha_c2 + end * stride_alpha_c3
        d_ptr = (alpha_d_ptr + start * stride_alpha_d2 + gap_start *
            stride_alpha_d3 + gap_end * stride_alpha_d4 + split *
            stride_alpha_d5)
        score = tl.load(c_ptr) + tl.load(d_ptr)
        max_idx = update_position(max_score, score, max_idx, tl.zeros((1,),
            dtype=tl.float32) + split + 3 * (L + 1))
        max_score = tl.maximum(score, max_score)
    span_score = tl.load(marginal_d + b_idx * stride_alpha_d1 + start *
        stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end *
        stride_alpha_d4 + end * stride_alpha_d5)
    tl.store(alpha_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 + 
        gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + end *
        stride_alpha_d5, max_score + span_score)
    tl.store(alpha_d + b_idx * stride_alpha_d1 + gap_start *
        stride_alpha_d2 + start * stride_alpha_d3 + gap_end *
        stride_alpha_d4 + end * stride_alpha_d5 + tl.arange(0, 1), max_idx)

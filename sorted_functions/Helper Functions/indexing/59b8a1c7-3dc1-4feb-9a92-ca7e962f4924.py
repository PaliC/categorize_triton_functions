import triton
import triton.language as tl
import torch

@triton.jit
def _kernel_argmax_merge_continuous(alpha_c, alpha_d, marginal_c,
    stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, stride_alpha_d1,
    stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5, B, w, L
    ):
    b_idx = tl.program_id(0)
    if b_idx >= B:
        return
    start = tl.program_id(1)
    end = start + w
    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (
        start + 1) * stride_alpha_c3
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start + 1
        ) * stride_alpha_c2 + end * stride_alpha_c3
    acc1 = tl.zeros((1,), dtype=tl.float32) - 1000000000.0
    max_idx = tl.zeros((1,), dtype=tl.float32) - 1
    for split in range(start + 1, start + w):
        left = tl.load(l_ptr)
        right = tl.load(r_ptr)
        merge = left + right
        max_idx = update_position(acc1, merge, max_idx, tl.zeros((1,),
            dtype=tl.float32) + split)
        acc1 = tl.maximum(merge, acc1)
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2
    for gap_start in range(start + 1, end - 1):
        for gap_end in range(gap_start + 1, end):
            ptr_c = (alpha_c + b_idx * stride_alpha_c1 + gap_start *
                stride_alpha_c2 + gap_end * stride_alpha_c3)
            ptr_d = (alpha_d + b_idx * stride_alpha_d1 + start *
                stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end *
                stride_alpha_d4 + end * stride_alpha_d5)
            cont = tl.load(ptr_c)
            disco = tl.load(ptr_d)
            merge = cont + disco
            max_idx = update_position(acc1, merge, max_idx, tl.zeros((1,),
                dtype=tl.float32) - (gap_start * L + gap_end))
            acc1 = tl.maximum(merge, acc1)
    tl.store(alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + 
        (start + w) * stride_alpha_c3 + tl.arange(0, 1), acc1 + tl.load(
        marginal_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (
        start + w) * stride_alpha_c3))
    tl.store(alpha_c + b_idx * stride_alpha_c1 + (start + w) *
        stride_alpha_c2 + start * stride_alpha_c3 + tl.arange(0, 1), max_idx)

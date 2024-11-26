import triton
import triton.language as tl
import torch

@triton.jit
def _kernel_inside_merge_continuous(alpha_c, alpha_d, tmp_merge,
    tmp_merge_normalized, tmp_normalizer, stride_alpha_c1, stride_alpha_c2,
    stride_alpha_c3, stride_alpha_d1, stride_alpha_d2, stride_alpha_d3,
    stride_alpha_d4, stride_alpha_d5, stride_tmp_merge1, stride_tmp_merge2,
    stride_tmp_merge_normalized1, r1, r2, r3, r4, b, n, w, L, BLOCK_R1:
    'tl.constexpr', BLOCK_R2: 'tl.constexpr'):
    b_idx = tl.program_id(0)
    start = tl.program_id(1)
    end = start + w
    if b_idx >= b:
        return
    offset_r = tl.arange(0, BLOCK_R1)
    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (
        start + 1) * stride_alpha_c3 + offset_r
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start + 1
        ) * stride_alpha_c2 + end * stride_alpha_c3 + r1 + offset_r
    acc1 = tl.zeros((BLOCK_R1,), dtype=tl.float32) - 1000000000.0
    mask = tl.arange(0, BLOCK_R1) < r1
    mask2 = tl.arange(0, BLOCK_R2) < r2
    for _ in range(0, w - 1):
        left = tl.load(l_ptr, mask=mask, other=-1000000000.0)
        right = tl.load(r_ptr, mask=mask, other=-1000000000.0)
        merge = left + right
        acc1 = logaddexp(acc1, merge)
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2
    acc2 = tl.zeros((BLOCK_R2,), dtype=tl.float32) - 1000000000.0
    for gap_start in range(start + 1, end - 1):
        for gap_end in range(gap_start + 1, end):
            ptr_c = (alpha_c + b_idx * stride_alpha_c1 + gap_start *
                stride_alpha_c2 + gap_end * stride_alpha_c3 + 2 * r1 + tl.
                arange(0, BLOCK_R2))
            ptr_d = (alpha_d + b_idx * stride_alpha_d1 + start *
                stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end *
                stride_alpha_d4 + end * stride_alpha_d5 + tl.arange(0,
                BLOCK_R2))
            cont = tl.load(ptr_c, mask=mask2, other=-1000000000.0)
            disco = tl.load(ptr_d, mask=mask2, other=-1000000000.0)
            merge = cont + disco
            acc2 = logaddexp(acc2, merge)
    tl.store(tmp_merge + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + tl.arange(0, BLOCK_R1), acc1, mask=mask)
    tl.store(tmp_merge + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), acc2, mask=mask2)
    acc1_max = tl.max(acc1, 0)
    acc2_max = tl.max(acc2, 0)
    acc_max = tl.maximum(acc1_max, acc2_max)
    tl.store(tmp_normalizer + b_idx * stride_tmp_merge_normalized1 + start,
        acc_max)
    out1 = tl.exp(acc1 - acc_max)
    tl.store(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + tl.arange(0, BLOCK_R1), out1, mask=mask)
    out2 = tl.exp(acc2 - acc_max)
    tl.store(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start *
        stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), out2, mask=mask2)

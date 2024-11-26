import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_log_and_diagonal_copy(out, out_grad, alpha_c, stride_alpha_c1,
    stride_alpha_c2, stride_alpha_c3, stride_out0, stride_out1, batch, r,
    BLOCK_R1: 'tl.constexpr', w):
    b_idx = tl.program_id(0)
    if b_idx >= batch:
        return
    mask = tl.arange(0, BLOCK_R1) < r
    start = tl.program_id(1)
    x = tl.load(out + b_idx * stride_out0 + start * stride_out1 + tl.arange
        (0, BLOCK_R1), mask=mask, other=1)
    out_log = 1 / (x + 1e-09)
    do = tl.load(alpha_c + b_idx * stride_alpha_c1 + (start + w) *
        stride_alpha_c2 + start * stride_alpha_c3 + tl.arange(0, BLOCK_R1),
        mask=mask, other=0)
    do *= out_log
    tl.store(out_grad + b_idx * stride_out0 + start * stride_out1 + tl.
        arange(0, BLOCK_R1), do, mask=mask)

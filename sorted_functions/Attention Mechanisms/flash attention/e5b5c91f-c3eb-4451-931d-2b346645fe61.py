import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_preprocess_do_o_dot(Out, DO, Delta, stride_ob, stride_oh,
    stride_om, stride_dob, stride_doh, stride_dom, nheads, seqlen_q,
    seqlen_q_rounded, headdim, BLOCK_M: 'tl.constexpr', BLOCK_HEADDIM:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:,
        None] * stride_om + offs_d[None, :], mask=(offs_m[:, None] <
        seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:,
        None] * stride_dom + offs_d[None, :], mask=(offs_m[:, None] <
        seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

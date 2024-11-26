import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_do_attn_kernel(O, Do, De, stride_ob, stride_om, stride_oh,
    stride_dob, stride_dom, stride_doh, stride_deb, stride_deh, nheads,
    headdim, seqlen_q, BLOCK_M: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr'):
    off_q = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = off_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None
        ] * stride_om + offs_d[None, :]
    do_ptrs = Do + off_b * stride_dob + off_h * stride_doh + offs_m[:, None
        ] * stride_dom + offs_d[None, :]
    o = tl.load(o_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :
        ] < headdim), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None,
        :] < headdim), other=0.0)
    delta = tl.sum(o * do, axis=1)
    tl.store(De + (off_b * stride_deb + off_h * stride_deh + offs_m), delta,
        mask=offs_m < seqlen_q)

import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_preprocess(Out, DO, Delta, cu_seqlens_q, mid_batch, mid_start,
    stride_oz, stride_oh, stride_ok, stride_doz, stride_doh, stride_dok,
    stride_dz, stride_dh, BLOCK_M: 'tl.constexpr', D_HEAD: 'tl.constexpr'):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.load(mid_batch + off_z)
    off_m = tl.load(mid_start + off_z)
    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)
    lM = q_end - q_start
    offs_m = tl.arange(0, BLOCK_M) + off_m
    offs_k = tl.arange(0, D_HEAD)
    o_ptrs = Out + (offs_m[:, None] * stride_oz + off_h * stride_oh + 
        offs_k[None, :] * stride_ok)
    do_ptrs = DO + (offs_m[:, None] * stride_doz + off_h * stride_doh + 
        offs_k[None, :] * stride_dok)
    mask_m = offs_m < q_end
    o = tl.load(o_ptrs, mask=mask_m[:, None])
    do = tl.load(do_ptrs, mask=mask_m[:, None])
    delta = tl.sum(o * do, axis=1)
    d_ptrs = Delta + offs_m * stride_dz + off_h * stride_dh
    tl.store(d_ptrs, delta, mask=mask_m)

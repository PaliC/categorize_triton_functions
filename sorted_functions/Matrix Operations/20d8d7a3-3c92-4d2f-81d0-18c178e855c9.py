import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_preprocess(Out, DO, Delta, stride_oz, stride_oh, stride_om,
    stride_ok, stride_doz, stride_doh, stride_dom, stride_dok, stride_dz,
    stride_dh, stride_dm, M, BLOCK_M: 'tl.constexpr', D_HEAD:
    'tl.constexpr', DIVISIBLE_M: 'tl.constexpr'):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok
    if DIVISIBLE_M:
        o = tl.load(o_ptrs)
        do = tl.load(do_ptrs)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
    delta = tl.sum(o * do, axis=1)
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)

import triton
import triton.language as tl
import torch

@triton.jit
def parallel_based_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t,
    s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr',
    BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK:
    'tl.constexpr', DV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_based_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq,
        s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL
        =BTL, BTS=BTS, BK=BK, BV=BV, DK=DK, DV=DV)
    tl.debug_barrier()
    _parallel_based_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk,
        dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale,
        BTL, BTS, BK, BV, DK, DV)

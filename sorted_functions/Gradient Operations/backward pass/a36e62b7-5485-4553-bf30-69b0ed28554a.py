import triton
import triton.language as tl
import torch

@triton.jit
def parallel_rebased_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_k_h, s_k_t,
    s_k_d, s_v_h, s_v_t, s_v_d, scale, B: 'tl.constexpr', H: 'tl.constexpr',
    T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL:
    'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'
    ):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq,
        s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, scale, B=B, H=H, T=T, K=K,
        V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV)
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk,
        dv, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, scale, B=B, H=H, T=T,
        K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV)

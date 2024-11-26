import triton
import triton.language as tl
import torch

@triton.heuristics({'NV': lambda args: triton.cdiv(args['V'], args['BV'])})
@triton.jit
def parallel_simple_gla_bwd_kernel(q, k, v, g, do, dq, dk, dv, dg, s_k_h,
    s_k_t, s_v_h, s_v_t, scale, B: 'tl.constexpr', H: 'tl.constexpr', T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BS: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NV: 'tl.constexpr'):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    parallel_simple_gla_bwd_kernel_dq(i_bh, i_t, i_k, i_v, q, k, v, g, do,
        dq, dg, s_k_h, s_k_t, s_v_h, s_v_t, scale, B=B, H=H, T=T, K=K, V=V,
        BT=BT, BS=BS, BK=BK, BV=BV)
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(i_bh, i_t, i_k, i_v, q, k, v, g, do,
        dk, dv, dg, s_k_h, s_k_t, s_v_h, s_v_t, scale, B, H, T, K, V, BT,
        BS, BK, BV)

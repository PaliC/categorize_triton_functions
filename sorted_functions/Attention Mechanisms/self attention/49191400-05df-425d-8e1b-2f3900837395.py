import triton
import triton.language as tl
import torch

@triton.heuristics({'NV': lambda args: triton.cdiv(args['V'], args['BV'])})
@triton.jit
def parallel_retention_bwd_kernel(q, k, v, do, dq, dk, dv, scale, B:
    'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr',
    V: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NV: 'tl.constexpr'):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_h = i_bh % H
    parallel_retention_bwd_kernel_dq(i_bh, i_t, i_k, i_v, i_h, k, v, do, dq,
        scale, B=B, H=H, T=T, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV)
    tl.debug_barrier()
    parallel_retention_bwd_kernel_dkv(i_bh, i_t, i_k, i_v, i_h, q, k, v, do,
        dk, dv, scale, B, H, T, K, V, BT, BS, BK, BV)

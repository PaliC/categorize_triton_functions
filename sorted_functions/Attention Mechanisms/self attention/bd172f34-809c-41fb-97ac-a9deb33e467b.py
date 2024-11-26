import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM'], 'EVEN_V_HEADDIM': lambda args: args['v_headdim'] ==
    args['V_BLOCK_HEADDIM']})
@triton.jit
def _bwd_sampled_col_kernel(Q, K, V, DO, DQ, DK, DV, LSE, D, softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm, stride_dkb, stride_dkh, stride_dkn,
    stride_dvb, stride_dvh, stride_dvn, nheads, seqlen_q, headdim,
    v_headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BLOCK_HEADDIM:
    'tl.constexpr', V_BLOCK_HEADDIM: 'tl.constexpr', EVEN_HEADDIM:
    'tl.constexpr', EVEN_V_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    Q += off_b * stride_qb + off_h * stride_qh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    D += off_hb * seqlen_q
    LSE += off_hb * seqlen_q
    start_n = tl.program_id(0)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_vd = tl.arange(0, V_BLOCK_HEADDIM)
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] *
        stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] *
        stride_vn + offs_vd[None, :])
    dv = tl.zeros([BLOCK_N, V_BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_HEADDIM:
        k = tl.load(k_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    if EVEN_V_HEADDIM:
        v = tl.load(v_ptrs)
    else:
        v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
    for start_m in range(0, seqlen_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :])
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        do_ptrs = DO + (offs_m_curr[:, None] * stride_dom + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        dv += tl.dot(tl.trans(p), do)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        Di = tl.load(D + offs_m_curr)
        ds = p * (dp - Di[:, None]) * softmax_scale
        dk += tl.dot(tl.trans(ds), q)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        dq_ptrs = DQ + (offs_m_curr[:, None] * stride_dqm + offs_d[None, :])
        dq = tl.dot(ds, k)
        if EVEN_HEADDIM:
            tl.atomic_add(dq_ptrs, dq)
        else:
            tl.atomic_add(dq_ptrs, dq, mask=offs_d[None, :] < headdim)
    dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + (offs_n[:,
        None] * stride_dvn + offs_vd[None, :])
    dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + (offs_n[:,
        None] * stride_dkn + offs_d[None, :])
    dk += tl.load(dk_ptrs)
    dv += tl.load(dv_ptrs)
    _bwd_store_dx(dk_ptrs, dk, offs_d, headdim, even_headdim=EVEN_HEADDIM)
    _bwd_store_dx(dv_ptrs, dv, offs_vd, v_headdim, even_headdim=EVEN_V_HEADDIM)
    return

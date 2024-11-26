import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM']})
@triton.jit
def _bwd_attn_kernel(Q, K, V, B, Do, L, D, softmax_scale, stride_qb,
    stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb,
    stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_dob,
    stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm, stride_dkb,
    stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn, stride_lb,
    stride_lh, seqlen_q, seqlen_k, headdim, nheads, Dq, Dk, Dv, HAVE_BIAS:
    'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_M: 'tl.constexpr',
    EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    off_n, off_b, off_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_n = off_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_m = tl.arange(0, BLOCK_M)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] *
        stride_qm + offs_d[None, :]))
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] *
        stride_kn + offs_d[None, :]))
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] *
        stride_vn + offs_d[None, :]))
    if HAVE_BIAS:
        b_ptrs = B + (off_b * stride_bb + off_h * stride_bh + (offs_m[:,
            None] * stride_bm + offs_n[None, :]))
    dq_ptrs = Dq + (off_b * stride_dqb + off_h * stride_dqh + (offs_m[:,
        None] * stride_dqm + offs_d[None, :]))
    dk_ptrs = Dk + (off_b * stride_dkb + off_h * stride_dkh + (offs_n[:,
        None] * stride_dkn + offs_d[None, :]))
    dv_ptrs = Dv + (off_b * stride_dvb + off_h * stride_dvh + (offs_n[:,
        None] * stride_dvn + offs_d[None, :]))
    do_ptrs = Do + (off_b * stride_dob + off_h * stride_doh + (offs_m[:,
        None] * stride_dom + offs_d[None, :]))
    lse_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_m)
    del_ptrs = D + (off_b * stride_lb + off_h * stride_lh + offs_m)
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :
        ] < headdim), other=0.0)
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :
        ] < headdim), other=0.0)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    for start_m in range(0, seqlen_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        m_loop_offs = start_m + offs_m
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs + start_m)
        elif EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=m_loop_offs[:, None] < seqlen_q, other=0.0
                )
        else:
            q = tl.load(q_ptrs + start_m, mask=(m_loop_offs[:, None] <
                seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.dot(q, k.T)
        if not EVEN_N:
            qk += tl.where(offs_n[None, :] < seqlen_k, 0, float('-inf'))
        l = tl.load(lse_ptrs + start_m, mask=m_loop_offs < seqlen_q, other=0.0
            )[:, None]
        if HAVE_BIAS:
            if EVEN_N & EVEN_M:
                b = tl.load(b_ptrs + start_m)
            else:
                b = tl.load(b_ptrs + start_m, mask=(m_loop_offs[:, None] <
                    seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0)
            qk = qk * softmax_scale + b
            p = tl.exp(qk - l)
        else:
            p = tl.exp(qk * softmax_scale - l)
        if EVEN_N & EVEN_HEADDIM:
            do = tl.load(do_ptrs + start_m)
        else:
            do = tl.load(do_ptrs + start_m, mask=(m_loop_offs[:, None] <
                seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        dv = dv + tl.dot(p.T, do)
        dp = tl.dot(do, v.T)
        di = tl.load(del_ptrs + start_m, mask=m_loop_offs < seqlen_q, other=0.0
            )
        ds = p * (dp - di[:, None]) * softmax_scale
        foqi = tl.dot(ds.T, q)
        dk += foqi
        dq = tl.dot(ds, k)
        pq = tl.load(dq_ptrs, mask=(m_loop_offs[:, None] < seqlen_q) & (
            offs_d[None, :] < headdim), other=0.0, eviction_policy='evict_last'
            )
        res = dq + pq
        tl.store(dq_ptrs, value=res, mask=(m_loop_offs[:, None] < seqlen_q) &
            (offs_d[None, :] < headdim), eviction_policy='evict_last')
        epq = tl.load(dq_ptrs, mask=(m_loop_offs[:, None] < seqlen_q) & (
            offs_d[None, :] < headdim), other=0.0, eviction_policy='evict_last'
            )
        tl.device_print('epq', epq)
    tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None,
        :] < headdim))
    tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None,
        :] < headdim))

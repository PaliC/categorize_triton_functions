import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM']})
@triton.jit
def _bwd_attn_kernel(Q, K, V, B, Do, L, D, softmax_scale: 'float',
    stride_qb: 'int', stride_qm: 'int', stride_qh: 'int', stride_kb: 'int',
    stride_kn: 'int', stride_kh: 'int', stride_vb: 'int', stride_vn: 'int',
    stride_vh: 'int', stride_bb: 'int', stride_bh: 'int', stride_bm: 'int',
    stride_dob: 'int', stride_dom: 'int', stride_doh: 'int', stride_dqb:
    'int', stride_dqm: 'int', stride_dqh: 'int', stride_dkb: 'int',
    stride_dkn: 'int', stride_dkh: 'int', stride_dvb: 'int', stride_dvn:
    'int', stride_dvh: 'int', stride_lb: 'int', stride_lh: 'int', seqlen_q:
    'int', seqlen_k: 'int', headdim: 'int', nheads: 'int', Dq: 'chex.Array',
    Dk: 'chex.Array', Dv: 'chex.Array', HAVE_BIAS: 'tl.constexpr',
    BIAS_SINGLE_HEAD: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_M:
    'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr',
    BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_n, off_bh = tl.program_id(0), tl.program_id(2)
    softmax_scale = softmax_scale
    off_h = off_bh % nheads
    off_b = off_bh // nheads
    offs_qm = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    l_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_qm)
    d_ptrs = D + (off_b * stride_lb + off_h * stride_lh + offs_qm)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + (offs_qm[:, None
        ] * stride_qm + offs_d[None, :])
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] *
        stride_kn + offs_d[None, :])
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] *
        stride_vn + offs_d[None, :])
    do_ptrs = Do + (off_b * stride_dob + off_h * stride_doh) + (offs_qm[:,
        None] * stride_dom + offs_d[None, :])
    dq_ptrs = Dq + (off_b * stride_dqb + off_h * stride_dqh) + (offs_qm[:,
        None] * stride_dqm + offs_d[None, :])
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h
        b_ptrs = B + (off_b * stride_bb + bias_h_pos * stride_bh) + (
            offs_qm[:, None] * stride_bm + offs_n[None, :])
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :
        ] < headdim), other=0.0)
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :
        ] < headdim), other=0.0)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(0, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (
            offs_d[None, :] < headdim), other=0.0)
        qk = tl.dot(q, k.T) * softmax_scale
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float('-inf'))
        if HAVE_BIAS:
            bias = tl.load(b_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) &
                (offs_n[None, :] < seqlen_k), other=0.0)
            qk = qk + bias
        lse_i = tl.load(l_ptrs + start_m, mask=offs_m_curr < seqlen_q,
            other=0.0)
        p = tl.exp(qk - lse_i[:, None])
        do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (
            offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(p.T, do)
        dp = tl.dot(do, v.T)
        Di = tl.load(d_ptrs + start_m, mask=offs_m_curr < seqlen_q, other=0.0)
        ds = p * (dp - Di[:, None]) * softmax_scale
        dk += tl.dot(ds.T, q)
        dq = tl.dot(ds, k)
        tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) &
            (offs_d[None, :] < headdim))
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if HAVE_BIAS:
            b_ptrs += BLOCK_M * stride_bm
    dv_ptrs = Dv + (off_b * stride_dvb + off_h * stride_dvh) + (offs_n[:,
        None] * stride_dvn + offs_d[None, :])
    dk_ptrs = Dk + (off_b * stride_dkb + off_h * stride_dkh) + (offs_n[:,
        None] * stride_dkn + offs_d[None, :])
    tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None,
        :] < headdim))
    tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None,
        :] < headdim))

import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM']})
@triton.jit
def _bwd_attention_kernel(Q, K, V, B, Do, L, D, softmax_scale:
    'tl.constexpr', stride_qb, stride_qm, stride_qh, stride_qg, stride_kb,
    stride_kn, stride_kh, stride_vb, stride_vn, stride_vh, stride_bb,
    stride_bh, stride_bg, stride_bm, stride_dob, stride_dom, stride_doh,
    stride_dog, stride_dqb, stride_dqm, stride_dqh, stride_dqg, stride_dkb,
    stride_dkn, stride_dkh, stride_dvb, stride_dvn, stride_dvh, stride_lb,
    stride_lh, stride_lg, seqlen_q, seqlen_k, headdim, num_kv_heads,
    num_groups, Dq, Dk, Dv, HAVE_BIAS: 'tl.constexpr', BIAS_SINGLE_HEAD:
    'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_M: 'tl.constexpr',
    EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_n, off_bh, off_gp = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    softmax_scale = softmax_scale
    off_h = off_bh % num_kv_heads
    off_b = off_bh // num_kv_heads
    offs_qm = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    l_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_qm + off_gp *
        stride_lg)
    d_ptrs = D + (off_b * stride_lb + off_h * stride_lh + offs_qm + off_gp *
        stride_lg)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh + off_gp * stride_qg
        ) + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] *
        stride_kn + offs_d[None, :])
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] *
        stride_vn + offs_d[None, :])
    do_ptrs = Do + (off_b * stride_dob + off_h * stride_doh + off_gp *
        stride_dog) + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = Dq + (off_b * stride_dqb + off_h * stride_dqh + off_gp *
        stride_dqg) + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = (0 if BIAS_SINGLE_HEAD else off_h *
            stride_bh + off_gp * stride_bg)
        b_ptrs = B + (off_b * stride_bb + bias_h_pos) + (offs_qm[:, None] *
            stride_bm + offs_n[None, :])
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

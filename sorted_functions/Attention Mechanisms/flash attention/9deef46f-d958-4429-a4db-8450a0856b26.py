import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM']})
@triton.jit
def _fwd_kernel(Q, K, V, Bias, Out, M_in, Lse_in, O_in, Lse, M_out, TMP,
    softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh,
    stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh,
    stride_bm, stride_ob, stride_oh, stride_om, nheads, seqlen_q, seqlen_k,
    seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', BLOCK_HEADDIM:
    'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_N: 'tl.constexpr',
    EVEN_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] *
        stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] *
        stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] *
        stride_vn + offs_d[None, :])
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:,
            None] * stride_bm + offs_n[None, :])
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lin_ptrs = Lse_in + off_hb * seqlen_q_rounded + offs_m
    acc_o_ptrs = O_in + off_b * stride_qb + off_h * stride_qh + (offs_m[:,
        None] * stride_qm + offs_d[None, :])
    lse_i = tl.load(lin_ptrs)
    m_ptrs = M_in + off_hb * seqlen_q_rounded + offs_m
    m_i = tl.load(m_ptrs)
    acc_o = tl.load(acc_o_ptrs)
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[
            None, :] < headdim), other=0.0)
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) *
        BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None,
                    :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n +
                offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float
                ('-inf'))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 
                0, float('-inf'))
        if BIAS_TYPE != 'none':
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=start_n + offs_n <
                        seqlen_k, other=0.0)
                bias = bias[None, :]
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(offs_m[:, None] <
                        seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k
                        ), other=0.0)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None,
                    :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=((start_n +
                offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0)
        p = p
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    m_ptrs = M_out + off_hb * seqlen_q_rounded + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:,
        None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (
            offs_d[None, :] < headdim))

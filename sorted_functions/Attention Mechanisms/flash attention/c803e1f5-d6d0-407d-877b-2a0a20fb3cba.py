import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM'], 'EVEN_V_HEADDIM': lambda args: args['v_headdim'] ==
    args['V_BLOCK_HEADDIM']})
@triton.jit
def _fwd_hyper_kernel(Q, K, V, q_sort_idx, k_sort_idx, Out, Lse,
    softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh,
    stride_kn, stride_vb, stride_vh, stride_vn, stride_q_sort_idxb,
    stride_q_sort_idxh, stride_q_sort_idxm, stride_k_sort_idxb,
    stride_k_sort_idxh, stride_k_sort_idxn, stride_ob, stride_oh, stride_om,
    nheads, block_size, sample_size, seqlen_k, seqlen_q, headdim, v_headdim,
    smooth_block, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BLOCK_HEADDIM:
    'tl.constexpr', V_BLOCK_HEADDIM: 'tl.constexpr', EVEN_HEADDIM:
    'tl.constexpr', EVEN_V_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_vd = tl.arange(0, V_BLOCK_HEADDIM)
    q_idx_ptrs = (q_sort_idx + off_b * stride_q_sort_idxb + off_h *
        stride_q_sort_idxh + offs_m * stride_q_sort_idxm)
    q_idx = tl.load(q_idx_ptrs)
    k_sort_idx += off_b * stride_k_sort_idxb + off_h * stride_k_sort_idxh
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, V_BLOCK_HEADDIM], dtype=tl.float32)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (q_idx[:, None] *
        stride_qm + offs_d[None, :])
    if EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    block_id = start_m // block_size
    block_offs = seqlen_k + start_m % block_size * BLOCK_N - (block_size - 1
        ) * BLOCK_N // 2
    end_n = tl.minimum((block_id + 1) * BLOCK_N * block_size, seqlen_k)
    for start_n in range(block_id * BLOCK_N * block_size, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if smooth_block:
            k_idx_ptrs = (start_n + block_offs + offs_n
                ) * stride_k_sort_idxn % seqlen_k
        else:
            k_idx_ptrs = (start_n + offs_n) * stride_k_sort_idxn
        k_idx = tl.load(k_sort_idx + k_idx_ptrs)
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (k_idx[:, None
            ] * stride_kn + offs_d[None, :])
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (k_idx[:, None
            ] * stride_vn + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        p = p
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    for col_block in range(0, sample_size):
        curr_offs_n = col_block * BLOCK_N * stride_kn + offs_n
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (curr_offs_n[:,
            None] * stride_kn + offs_d[None, :])
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (curr_offs_n[:,
            None] * stride_vn + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        p = p
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    lse_ptrs = Lse + off_hb * seqlen_q + q_idx
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (q_idx[:, None
        ] * stride_om + offs_vd[None, :])
    tl.store(lse_ptrs, lse_i)
    if EVEN_V_HEADDIM:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_vd[None, :] < v_headdim)

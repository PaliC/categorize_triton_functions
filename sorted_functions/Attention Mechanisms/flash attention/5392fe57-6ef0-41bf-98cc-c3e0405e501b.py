import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, m_i, d_i, q, k_ptrs, v_ptrs, k_seq_stride,
    v_seq_stride, offs_m, qk_scale, n_size, causal_mask, BLOCK_M_SIZE:
    'tl.constexpr', BLOCK_N_SIZE: 'tl.constexpr', fp8_v: 'tl.constexpr'):
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)
    for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
        block_n_start_idx = tl.multiple_of(block_n_start_idx, BLOCK_N_SIZE)
        block_n_offs = block_n_start_idx + n_range_offs
        k_mask = block_n_offs[:, None] < n_size
        k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask,
            other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if causal_mask:
            offs_k = block_n_offs
            mask = offs_m[:, None] >= offs_k[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -100000000.0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        d_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask,
            other=0.0)
        p = p
        acc = tl.dot(p, v, acc)
        m_i = m_ij
    return acc, d_i

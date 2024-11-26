import triton
import triton.language as tl
import torch

@triton.jit
def _single_query_cached_kv_attention_v1(out, q, k_cache, v_cache,
    head_mapping, scale, block_tables, seq_lens, max_num_blocks_per_seq,
    stride_qm, stride_qn, stride_om, stride_on, stride_km, stride_kn,
    stride_kk, SLOT_SIZE: 'tl.constexpr', HEAD_SIZE: 'tl.constexpr'):
    head_idx = tl.program_id(axis=0)
    token_idx = tl.program_id(axis=1)
    kv_head_idx = tl.load(head_mapping + head_idx)
    offs_q = token_idx * stride_qm + head_idx * stride_qn + tl.arange(0,
        HEAD_SIZE)
    q = tl.load(q + offs_q)
    q = q * scale
    seq_len = tl.load(seq_lens + token_idx)
    qkv = tl.zeros([SLOT_SIZE, HEAD_SIZE], dtype=tl.float32)
    m_prev = tl.zeros([1, 1], tl.float32) - float('inf')
    d_prev = tl.zeros([1, 1], tl.float32)
    slot_offs = tl.arange(0, SLOT_SIZE)
    head_size_offs = tl.arange(0, HEAD_SIZE)
    block_base_ptrs = block_tables + token_idx * max_num_blocks_per_seq
    kv_base_offs = kv_head_idx * stride_kn + slot_offs[:, None
        ] * stride_kk + head_size_offs[None, :]
    for i in range(0, tl.cdiv(seq_len, SLOT_SIZE)):
        block_idx = tl.load(block_base_ptrs + i)
        mask = (slot_offs[:, None] < seq_len - i * SLOT_SIZE) & (head_size_offs
            [None, :] < HEAD_SIZE)
        kv_offs = block_idx * stride_km + kv_base_offs
        k = tl.load(k_cache + kv_offs, mask=mask, other=0.0)
        v = tl.load(v_cache + kv_offs, mask=mask, other=0.0)
        x_i = tl.sum(q[None, :] * k, axis=1)[:, None]
        x_i = tl.where(slot_offs[:, None] < seq_len - i * SLOT_SIZE, x_i,
            float('-inf'))
        m_i = tl.maximum(m_prev, tl.max(x_i, axis=0))
        d_i = d_prev * tl.exp(m_prev - m_i) + tl.sum(tl.exp(x_i - m_i), axis=0)
        qkv = qkv * (d_prev * tl.exp(m_prev - m_i) / d_i) + tl.exp(x_i - m_i
            ) / d_i * v
        m_prev = m_i
        d_prev = d_i
    offs_q = token_idx * stride_om + head_idx * stride_on + tl.arange(0,
        HEAD_SIZE)
    tl.store(out + offs_q, tl.sum(qkv, axis=0))

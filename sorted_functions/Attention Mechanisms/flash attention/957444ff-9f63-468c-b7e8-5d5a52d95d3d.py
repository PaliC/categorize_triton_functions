import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
    block_index_q, softmax_scale, BLOCK_SIZE_Q: 'tl.constexpr',
    BLOCK_SIZE_KV: 'tl.constexpr', STAGE: 'tl.constexpr', offs_q:
    'tl.constexpr', offs_kv: 'tl.constexpr', SEQ_LEN: 'tl.constexpr'):
    if STAGE == 1:
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1
            ) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        lo, hi = 0, SEQ_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        if STAGE == 2:
            mask = offs_q[:, None] >= start_kv + offs_kv[None, :]
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1000000.0)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        V_block = tl.load(V_block_ptr)
        P_block = P_block
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i

import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m,
    qk_scale, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', STAGE: 'tl.constexpr', offs_m: 'tl.constexpr',
    offs_n: 'tl.constexpr', N_CTX: 'tl.constexpr'):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if STAGE != 1:
            k = tl.load(K_block_ptr, boundary_check=(0, 1))
        else:
            k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE != 1:
            n_ctx_mask = tl.where((offs_m[:, None] < N_CTX) & (start_n +
                offs_n[None, :] < N_CTX), 0, float('-inf'))
            qk += n_ctx_mask
        qk += tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= start_n + offs_n[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, float('-inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        if STAGE != 1:
            v = tl.load(V_block_ptr, boundary_check=(0, 1))
        else:
            v = tl.load(V_block_ptr)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

import triton
import triton.language as tl
import torch

@triton.jit
def _prob_fwd_kernel(Q, K, LSE, nheads, seqlen_q, seqlen_k, BLOCK_HEADDIM:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q,
                other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
            qk += tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where((start_n + offs_n)[None, :] < seqlen_k, p, 0.0)
        lse_i = tl.exp(m_i - m_ij) * lse_i + tl.sum(p, 1)
        m_i = m_ij
    lse_i = m_i + tl.log(lse_i)
    lse_i = tl.where(offs_m < seqlen_q, lse_i, 0.0)
    tl.store(LSE + offs_m, lse_i)

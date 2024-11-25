import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, Out, sqz, sqh, sqm, sqd, skz, skh, skn,
    skd, svz, svh, svn, svd, soz, soh, som, sod, L, M, Z, H, N_CTX_Q,
    N_CTX_KV, BLOCK: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    N_PREFIX_Q: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    BLOCK_M: 'tl.constexpr' = BLOCK
    BLOCK_N: 'tl.constexpr' = BLOCK
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_real = (start_m + N_PREFIX_Q) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd
    offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)
    for start_n in range(0, N_PREFIX_Q + start_m):
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
        qk += tl.dot(q_vals, k_vals, allow_tf32=False)
        qk *= sm_scale
        qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float(
            '-inf'))
        landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] == 
            BLOCK_N - 1, qk, float('-inf')), 1)
        normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1,
            float('-inf'), qk)
        normal_m = tl.max(normal_qk, 1)
        normal_p = tl.exp(normal_qk - normal_m[:, None])
        normal_denom = tl.sum(normal_p, 1)
        m_curr = tl.maximum(landmark_qk, m_prev)
        m_curr_ = m_curr
        l_prev *= tl.exp(m_prev - m_curr_)
        landmark_p = tl.exp(landmark_qk - m_curr_)
        l_curr = landmark_p + l_prev
        l_rcp = 1.0 / l_curr
        landmark_p *= l_rcp
        acc *= (l_prev * l_rcp)[:, None]
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
        acc += tl.dot(landmark_p[:, None] * normal_p / normal_denom[:, None
            ], v_vals, allow_tf32=False)
        l_prev = l_curr
        m_prev = m_curr
        offs_n += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_v += BLOCK_N * svn
    k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
    qk += tl.dot(q_vals, k_vals, allow_tf32=False)
    qk *= sm_scale
    qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float('-inf'))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    m_curr_ = m_curr
    l_prev *= tl.exp(m_prev - m_curr_)
    p = tl.exp(qk - m_curr_[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = p
    v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
    acc += tl.dot(p, v_vals, allow_tf32=False)
    l_prev = l_curr
    m_prev = m_curr
    offs_L = off_hz * N_CTX_Q + offs_m
    offs_M = off_hz * N_CTX_Q + offs_m
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)
    offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)

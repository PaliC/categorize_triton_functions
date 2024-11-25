import triton
import triton.language as tl
import torch

@triton.jit
def triton_dense_fwd_kernel(Q, K, V, seqlens, sm_scale, Out, stride_qz,
    stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
    stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz,
    stride_oh, stride_om, stride_ok, Z, H, N_CTX, BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', dtype:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return
    qo_offset = off_hz // H * stride_qz + off_hz % H * stride_qh
    kv_offset = off_hz // H * stride_kz + off_hz % H * stride_kh
    Q_block_ptr = tl.make_block_ptr(base=Q + qo_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + kv_offset, shape=(BLOCK_DMODEL,
        N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape
        =(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + kv_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = q * qk_scale
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    m_mask = offs_m[:, None] < seqlen
    for start_n in range(lo, hi, BLOCK_N):
        n_mask = start_n + offs_n[None, :] <= offs_m[:, None]
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    O_block_ptr = tl.make_block_ptr(base=Out + qo_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_om, stride_ok), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(O_block_ptr, acc, mask=m_mask)

import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, L, Out, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
    stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om,
    stride_on, Z, H, N_CTX, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(
        BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0
        ), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_vk, stride_vn), offsets=(0, 0),
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
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk,
                float('-inf'))
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
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    O_block_ptr = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(O_block_ptr, acc)

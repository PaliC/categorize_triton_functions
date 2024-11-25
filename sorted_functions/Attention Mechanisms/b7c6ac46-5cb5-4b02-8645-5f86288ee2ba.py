import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, L, Out, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om,
    stride_on, Z, H, N_CTX, Z_H_N_CTX, BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_CAUSAL:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm
    K_block_ptr = tl.make_block_ptr(base=K, shape=(BLOCK_DMODEL, Z_H_N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, vk_offset), block_shape
        =(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(vk_offset, 0), block_shape
        =(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :
        ] * stride_qk
    q = tl.load(Q_ptrs)
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
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    O_block_ptr = tl.make_block_ptr(base=Out, shape=(Z_H_N_CTX,
        BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(vk_offset +
        start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(
        1, 0))
    tl.store(O_block_ptr, acc)

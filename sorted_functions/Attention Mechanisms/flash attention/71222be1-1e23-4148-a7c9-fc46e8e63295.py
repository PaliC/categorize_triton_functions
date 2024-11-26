import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
    stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om,
    stride_on, Z, H, N_CTX, HEAD_DIM: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', STAGE: 'tl.constexpr'):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX,
        HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    v_order: 'tl.constexpr' = (1, 0)
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX,
        HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=v_order)
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM,
        N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape
        =(HEAD_DIM, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX,
        HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = tl.load(Q_block_ptr)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
            V_block_ptr, start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, 4 -
            STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
            V_block_ptr, start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, 2,
            offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc)

import triton
import triton.language as tl
import torch

@triton.jit
def _triton_attn_fwd(Q, K, V, sm_scale, Out, stride_qz, stride_qh,
    stride_qm, stride_qk, stride_kz, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vh, stride_vm, stride_vk, stride_oz, stride_oh,
    stride_om, stride_ok, Z, H, N_CTX, POWER_OF_2_N_CTX: 'tl.constexpr',
    BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', STAGE: 'tl.constexpr', GROUPS: 'tl.constexpr', ORDER_12:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_g = tl.program_id(2)
    q_offset = off_z * stride_qz + (off_h * GROUPS + off_g) * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + (off_h * GROUPS + off_g) * stride_oh
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_vm, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(BLOCK_DMODEL,
        N_CTX), strides=(stride_kk, stride_km), offsets=(0, 0), block_shape
        =(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_om, stride_ok), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    if ORDER_12:
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
                V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL,
                BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX)
        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
                V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL,
                BLOCK_N, 2, offs_m, offs_n, N_CTX)
    else:
        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
                V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL,
                BLOCK_N, 2, offs_m, offs_n, N_CTX)
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
                V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL,
                BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc, boundary_check=(0, 1))

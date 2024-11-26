import triton
import triton.language as tl
import torch

@triton.heuristics({'IS_EVEN_M': lambda args: args['N_CTX'] % args[
    'BLOCK_M'] == 0, 'IS_EVEN_N': lambda args: args['NKV_CTX'] % args[
    'BLOCK_N'] == 0})
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, L, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
    stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om,
    stride_on, Z, H, H_KV, N_CTX, ROUND_CTX, NKV_CTX, sliding_window_offset,
    sliding_window_size, IS_EVEN_M: 'tl.constexpr', IS_EVEN_N:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', END: 'tl.constexpr', INIT: 'tl.constexpr',
    SLIDING_WINDOW: 'tl.constexpr', COMPLEMENT_SLIDING_WINDOW: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H // H_KV)
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_hkv * stride_kh
    v_offset = off_z * stride_vz + off_hkv * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(NKV_CTX,
        BLOCK_DMODEL), strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(BLOCK_DMODEL,
        NKV_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(ROUND_CTX,
        BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_ptrs = M + off_hz * ROUND_CTX + offs_m
    l_ptrs = L + off_hz * ROUND_CTX + offs_m
    if INIT:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    else:
        m_i = tl.load(m_ptrs)
        l_i = tl.load(l_ptrs)
        acc = tl.load(O_block_ptr)
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634
    if IS_EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
        V_block_ptr, start_m, qk_scale, NKV_CTX, sliding_window_offset,
        sliding_window_size, BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW,
        IS_EVEN_M, IS_EVEN_N, COMPLEMENT_SLIDING_WINDOW)
    if END:
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
    else:
        tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc)

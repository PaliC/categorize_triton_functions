import triton
import triton.language as tl
import torch

@triton.autotune(configs=_get_configs(), key=['N_CTX', 'H', 'Z'])
@triton.heuristics({'EVEN_CTX': lambda args: args['N_CTX'] % args['BLOCK_M'
    ] == 0})
@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, qkv_scale_ptr, out_scale_ptr, Out,
    stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
    stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, EVEN_CTX:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(
        BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0
        ), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qkv_scale = tl.load(qkv_scale_ptr)
    qk_scale = qkv_scale * qkv_scale * sm_scale * 1.44269504
    if EVEN_CTX:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_CTX:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option='zero'
                )
        qk = tl.dot(q, k, allow_tf32=False, out_dtype=tl.int32)
        qk_fp32 = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk_fp32, 1))
        p = tl.math.exp2(qk_fp32 - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        m_i = m_ij
        if EVEN_CTX:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero'
                )
        v = v * qkv_scale
        acc *= alpha[:, None]
        acc += tl.dot(p, v, allow_tf32=True)
        l_i = l_i * alpha + tl.sum(p, 1)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    out_scale = tl.load(out_scale_ptr)
    acc = tl.math.llrint(acc / (l_i[:, None] * out_scale))
    O_block_ptr = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    if EVEN_CTX:
        tl.store(O_block_ptr, acc)
    else:
        tl.store(O_block_ptr, acc, boundary_check=(0,))

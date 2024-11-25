import triton
import triton.language as tl
import torch

@triton.jit
def _triton_block_sparse_attn_fwd_kernel(Q, K, V, seqlens, sm_scale,
    block_index, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
    stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn,
    stride_vk, stride_oz, stride_oh, stride_om, stride_ok, Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW, BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', dtype: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    qo_offset = off_hz // H * stride_qz + off_hz % H * stride_qh
    kv_offset = off_hz // H * stride_kz + off_hz % H * stride_kh
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :
        ] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :
        ] * stride_ok
    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m
        ) * MAX_BLOCKS_PRE_ROW
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = q * qk_scale
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N,
        MAX_BLOCKS_PRE_ROW)
    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc, mask=m_mask)

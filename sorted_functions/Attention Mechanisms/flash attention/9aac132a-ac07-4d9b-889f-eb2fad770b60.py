import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner_ws(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, desc_k,
    desc_v, Q, qvk_offset, stride_kn, stride_vn, stride_vk, start_m,
    qk_scale, BLOCK_M: 'tl.constexpr', HEAD_DIM: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', STAGE: 'tl.constexpr', offs_m: 'tl.constexpr', offs_n:
    'tl.constexpr', N_CTX: 'tl.constexpr', fp8_v: 'tl.constexpr',
    ENABLE_TMA: 'tl.constexpr', LOOP_SCHEDULE: 'tl.constexpr'):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    if not ENABLE_TMA:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        with tl.async_task([0]):
            if ENABLE_TMA:
                k = tl._experimental_descriptor_load(desc_k, [start_n + 
                    qvk_offset // stride_kn, 0], [BLOCK_N, HEAD_DIM], Q.
                    dtype.element_ty)
            else:
                k = tl.load(K_block_ptr)
        with tl.async_task([1, 2]):
            if ENABLE_TMA:
                k = tl.trans(k)
            qk = tl.dot(q, k)
            if STAGE == 2:
                mask = offs_m[:, None] >= start_n + offs_n[None, :]
                qk = qk * qk_scale + tl.where(mask, 0, -1000000.0)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
        with tl.async_task([0]):
            if ENABLE_TMA:
                if fp8_v:
                    v = tl._experimental_descriptor_load(desc_v, [
                        qvk_offset // stride_vn, start_n], [HEAD_DIM,
                        BLOCK_N], Q.dtype.element_ty)
                else:
                    v = tl._experimental_descriptor_load(desc_v, [
                        qvk_offset // stride_vk + start_n, 0], [BLOCK_N,
                        HEAD_DIM], Q.dtype.element_ty)
            else:
                v = tl.load(V_block_ptr)
        with tl.async_task([1, 2]):
            if fp8_v:
                if ENABLE_TMA:
                    v = tl.trans(v)
                p = p
            else:
                p = p
            acc = tl.dot(p, v, acc)
            m_i = m_ij
        if not ENABLE_TMA:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

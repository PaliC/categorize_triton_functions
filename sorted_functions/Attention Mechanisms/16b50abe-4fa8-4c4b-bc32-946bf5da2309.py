import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_split_kv_kernel(Q, K, V, sm_scale, L, O, stride_qz, stride_qh,
    stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh,
    stride_os, stride_om, stride_ok, Z, H, M, N, P_SEQ, N_SPLIT_SIZE, S,
    num_groups, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', LARGER_M:
    'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    n_split_id = tl.program_id(1)
    off_zh = tl.program_id(2)
    off_h = off_zh % H
    off_z = off_zh // H
    off_hk = off_h // num_groups
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh + n_split_id * stride_os
    L += ((off_z * H + off_h) * S + n_split_id) * M
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k, tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 1.0, dtype=input_dtype), tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I)
    N_LEFT = n_split_id * N_SPLIT_SIZE
    N_RIGHT = tl.minimum(N_LEFT + N_SPLIT_SIZE, N)
    if IS_CAUSAL:
        hi = tl.minimum(N_RIGHT, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(N_LEFT, hi)
    else:
        hi = N_RIGHT
    offs_n_init = N_LEFT + offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] *
        stride_vn)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] *
        stride_kk)
    for start_n in range(N_LEFT, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier='.cg')
            v = tl.load(v_ptrs, cache_modifier='.cg')
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier='.cg')
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier='.cg')
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float('-inf'))
        if IS_CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    if IS_CAUSAL and LARGER_M:
        is_empty_line = offs_m + P_SEQ < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float('-inf'), m_i * sm_scale + tl.log(l_i)
            )
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i)
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier='.cg')
        tl.store(o_ptrs, acc, cache_modifier='.cg')
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier='.cg')
        tl.store(o_ptrs, acc, mask=mask_m[:, None], cache_modifier='.cg')

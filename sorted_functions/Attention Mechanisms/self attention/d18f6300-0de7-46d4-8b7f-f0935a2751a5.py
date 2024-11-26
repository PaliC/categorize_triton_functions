import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, B, sm_scale, L, ml, O, stride_qz, stride_qh,
    stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh,
    stride_os, stride_om, stride_lz, stride_lh, stride_ls, stride_lm,
    stride_bz, stride_bh, stride_bm, stride_bn, Z, H, M, N, P_SEQ, BLOCK_M:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    DIVISIBLE_N: 'tl.constexpr', HAS_BIAS: 'tl.constexpr', NUM_SPLITS:
    'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
    off_s = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    n_per_split = N // NUM_SPLITS
    split_n_start = off_s * n_per_split
    split_n_end = N if off_s + 1 == NUM_SPLITS else split_n_start + n_per_split
    log2e: 'tl.constexpr' = 1.4426950408889634
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh + off_s * stride_os
    if HAS_BIAS:
        B += off_z * stride_bz + off_h * stride_bh
    L += off_z * stride_lz + off_h * stride_lh + off_s * stride_ls
    ml += off_z * stride_lz + off_h * stride_lh + off_s * stride_ls
    offs_m = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :])
    l_ptrs = L + offs_m
    ml_ptrs = ml + offs_m
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    mask_m = offs_m < M
    q = tl.load(q_ptrs, cache_modifier='.cg')
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k, tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 1.0, dtype=input_dtype), tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I)
    offs_n_init = offs_n_base + split_n_start
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] *
        stride_vn)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] *
        stride_kk)
    if HAS_BIAS:
        bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n_init[None, :] *
            stride_bn)
    for start_n in range(split_n_start, split_n_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        mask_n = offs_n < N
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier='.cg')
            v = tl.load(v_ptrs, cache_modifier='.cg')
        else:
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier='.cg')
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier='.cg')
        if HAS_BIAS:
            b = tl.load(bias_ptrs, mask_m[:, None] & mask_n[None, :])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k) * sm_scale
        if HAS_BIAS:
            s += b
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        p = tl.math.exp2((s - m_i_new[:, None]) * log2e)
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        if HAS_BIAS:
            bias_ptrs += BLOCK_N * stride_bn
    acc = acc * (1.0 / l_i[:, None])
    l = l_i
    tl.store(l_ptrs, l, mask=mask_m, cache_modifier='.cg')
    tl.store(ml_ptrs, m_i, mask=mask_m, cache_modifier='.cg')
    tl.store(o_ptrs, acc, mask=mask_m[:, None], cache_modifier='.cg')

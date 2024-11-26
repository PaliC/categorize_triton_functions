import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, B, sm_scale, L, O, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om,
    stride_ok, stride_bz, stride_bh, stride_bm, stride_bn, Z, H, M, N,
    P_SEQ, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', IS_CAUSAL: 'tl.constexpr', LARGER_M: 'tl.constexpr',
    DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr', HAS_BIAS:
    'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    if HAS_BIAS:
        B += off_z * stride_bz + off_h * stride_bh
    L += (off_z * H + off_h) * M
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
    mask_m = offs_m < M
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier='.cg')
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier='.cg')
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k, tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 1.0, dtype=input_dtype), tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I)
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] *
        stride_vn)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] *
        stride_kk)
    if HAS_BIAS:
        bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n_init[None, :] *
            stride_bn)
    for start_n in range(0, hi, BLOCK_N):
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
            if DIVISIBLE_M and DIVISIBLE_N:
                b = tl.load(bias_ptrs)
            else:
                b = tl.load(bias_ptrs, mask_m[:, None] & mask_n[None, :])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k) * sm_scale
        if HAS_BIAS:
            s += b
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float('-inf'))
        if IS_CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
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
    if IS_CAUSAL and LARGER_M:
        is_empty_line = offs_m + P_SEQ < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float('-inf'), m_i + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i + tl.log(l_i)
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier='.cg')
        tl.store(o_ptrs, acc, cache_modifier='.cg')
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier='.cg')
        tl.store(o_ptrs, acc, mask=mask_m[:, None], cache_modifier='.cg')

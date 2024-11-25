import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q1, K1, Q2, K2, V, sm_scale, L, O, stride_q1z, stride_q1h,
    stride_q1m, stride_q1k, stride_k1z, stride_k1h, stride_k1n, stride_k1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k, stride_k2z, stride_k2h,
    stride_k2n, stride_k2k, stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok, Z, H, M, N, P_SEQ, w:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', LARGER_M:
    'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    input_dtype = Q1.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q1_ptrs = Q1 + (offs_m[:, None] * stride_q1m + offs_k[None, :] * stride_q1k
        )
    q2_ptrs = Q2 + (offs_m[:, None] * stride_q2m + offs_k[None, :] * stride_q2k
        )
    k1_ptrs = K1 + (offs_n_init[:, None] * stride_k1n + offs_k[None, :] *
        stride_k1k)
    k2_ptrs = K2 + (offs_n_init[:, None] * stride_k2n + offs_k[None, :] *
        stride_k2k)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] *
        stride_vk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        q1 = tl.load(q1_ptrs)
        q2 = tl.load(q2_ptrs)
    else:
        mask_m = offs_m < M
        q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
        q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
    I = tl.where(offs_k[:, None] == offs_k, tl.full((BLOCK_DMODEL,
        BLOCK_DMODEL), 1.0, dtype=input_dtype), tl.full((BLOCK_DMODEL,
        BLOCK_DMODEL), 0.0, dtype=input_dtype))
    q1 = tl.dot(q1, I)
    q2 = tl.dot(q2, I)
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        piecewise_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :] + w
        if DIVISIBLE_N:
            k1 = tl.load(k1_ptrs)
            k2 = tl.load(k2_ptrs)
            v = tl.load(v_ptrs)
        else:
            mask_n = offs_n < N
            k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
            k2 = tl.load(k2_ptrs, mask=mask_n[:, None])
            v = tl.load(v_ptrs, mask=mask_n[:, None])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(piecewise_mask, tl.dot(q2, tl.trans(k2)), tl.dot(q1,
            tl.trans(k1)))
        if not DIVISIBLE_N:
            s = tl.where(mask_n, s, float('-inf'))
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
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn
    if IS_CAUSAL and LARGER_M:
        is_empty_line = offs_m + P_SEQ < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l_i = tl.where(is_empty_line, float('-inf'), m_i * sm_scale + tl.
            log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l_i = m_i * sm_scale + tl.log(l_i)
    if DIVISIBLE_M:
        tl.store(l_ptrs, l_i)
        tl.store(o_ptrs, acc)
    else:
        tl.store(l_ptrs, l_i, mask=mask_m)
        tl.store(o_ptrs, acc, mask=mask_m[:, None])

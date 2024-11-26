import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_q_kernel(Q, K, V, B, sm_scale, DO, DQ, L, D, stride_qz, stride_qh,
    stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk, stride_bz, stride_bh,
    stride_bm, stride_bn, stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk, Z, H, M, N, P_SEQ,
    BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', CAUSAL: 'tl.constexpr', LARGER_M: 'tl.constexpr',
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
    if HAS_BIAS:
        B += off_z * stride_bz + off_h * stride_bh
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M
    DQ += off_z * stride_dqz + off_h * stride_dqh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] *
        stride_kk)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] *
        stride_vk)
    if HAS_BIAS:
        bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n_init[None, :] *
            stride_bn)
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        )
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        )
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m
    mask_m = offs_m < M
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < N
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k = tl.load(k_ptrs)
        else:
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k = tl.load(k_ptrs, mask=mask_n[:, None])
        if HAS_BIAS:
            if DIVISIBLE_M and DIVISIBLE_N:
                b = tl.load(bias_ptrs)
            else:
                b = tl.load(bias_ptrs, mask=mask_m[:, None] & mask_n[None, :])
        if not DIVISIBLE_N:
            valid_mask = mask_n
        if CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_BIAS:
            s += b
        p = tl.math.exp2((s - l[:, None]) * log2e)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_N:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        dq += tl.dot(ds, k)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        if HAS_BIAS:
            bias_ptrs += BLOCK_N * stride_bn
    dq *= sm_scale
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq)
    else:
        tl.store(dq_ptrs, dq, mask=mask_m[:, None])

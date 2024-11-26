import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_q_kernel(Q1, K1, Q2, K2, V, sm_scale, DO, DQ1, DQ2, L, D,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k, stride_k1z, stride_k1h,
    stride_k1n, stride_k1k, stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_k2z, stride_k2h, stride_k2n, stride_k2k, stride_vz, stride_vh,
    stride_vn, stride_vk, stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dq1z, stride_dq1h, stride_dq1m, stride_dq1k, stride_dq2z,
    stride_dq2h, stride_dq2m, stride_dq2k, Z, H, M, N, P_SEQ, w:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', CAUSAL: 'tl.constexpr', LARGER_M:
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
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M
    DQ1 += off_z * stride_dq1z + off_h * stride_dq1h
    DQ2 += off_z * stride_dq2z + off_h * stride_dq2h
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
    dq1_ptrs = DQ1 + (offs_m[:, None] * stride_dq1m + offs_k[None, :] *
        stride_dq1k)
    dq2_ptrs = DQ2 + (offs_m[:, None] * stride_dq2m + offs_k[None, :] *
        stride_dq2k)
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        )
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m
    if DIVISIBLE_M:
        q1 = tl.load(q1_ptrs)
        q2 = tl.load(q2_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
    else:
        mask_m = offs_m < M
        q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
        q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)
    dq1 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dq2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k1 = tl.load(k1_ptrs)
            k2 = tl.load(k2_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
            k2 = tl.load(k2_ptrs, mask=mask_n[:, None])
        piecewise_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :] + w
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(piecewise_mask, tl.dot(q2, tl.trans(k2)), tl.dot(q1,
            tl.trans(k1)))
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_N:
            ds = tl.where(mask_n, ds, 0.0)
        if CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            ds = tl.where(causal_mask, ds, 0.0)
        ds2 = tl.where(piecewise_mask, ds, 0.0)
        ds1 = tl.where(piecewise_mask, 0.0, ds)
        dq1 += tl.dot(ds1, k1)
        dq2 += tl.dot(ds2, k2)
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn
    dq1 *= sm_scale
    dq2 *= sm_scale
    if DIVISIBLE_M:
        tl.store(dq1_ptrs, dq1)
        tl.store(dq2_ptrs, dq2)
    else:
        tl.store(dq1_ptrs, dq1, mask=mask_m[:, None])
        tl.store(dq2_ptrs, dq2, mask=mask_m[:, None])

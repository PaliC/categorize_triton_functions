import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kv_kernel(Q1, K1, Q2, K2, V, sm_scale, DO, DK1, DK2, DV, L, D,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k, stride_k1z, stride_k1h,
    stride_k1n, stride_k1k, stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_k2z, stride_k2h, stride_k2n, stride_k2k, stride_vz, stride_vh,
    stride_vn, stride_vk, stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dk1z, stride_dk1h, stride_dk1n, stride_dk1k, stride_dk2z,
    stride_dk2h, stride_dk2n, stride_dk2k, stride_dvz, stride_dvh,
    stride_dvn, stride_dvk, Z, H, M, N, P_SEQ, w: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    CAUSAL: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N:
    'tl.constexpr'):
    input_dtype = Q1.dtype.element_ty
    start_n = tl.program_id(0)
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
    DK1 += off_z * stride_dk1z + off_h * stride_dk1h
    DK2 += off_z * stride_dk2z + off_h * stride_dk2h
    DV += off_z * stride_dvz + off_h * stride_dvh
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = lo // BLOCK_M * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q1_ptrs = Q1 + (offs_m_init[:, None] * stride_q1m + offs_k[None, :] *
        stride_q1k)
    q2_ptrs = Q2 + (offs_m_init[:, None] * stride_q2m + offs_k[None, :] *
        stride_q2k)
    k1_ptrs = K1 + (offs_k[:, None] * stride_k1k + offs_n[None, :] * stride_k1n
        )
    k2_ptrs = K2 + (offs_k[:, None] * stride_k2k + offs_n[None, :] * stride_k2n
        )
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] *
        stride_dok)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk
        )
    dk1_ptrs = DK1 + (offs_n[:, None] * stride_dk1n + offs_k[None, :] *
        stride_dk1k)
    dk2_ptrs = DK2 + (offs_n[:, None] * stride_dk2n + offs_k[None, :] *
        stride_dk2k)
    if DIVISIBLE_N:
        k1 = tl.load(k1_ptrs)
        k2 = tl.load(k2_ptrs)
        v = tl.load(v_ptrs)
    else:
        mask_n = offs_n < N
        k1 = tl.load(k1_ptrs, mask=mask_n[None, :])
        k2 = tl.load(k2_ptrs, mask=mask_n[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None])
    dk1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        if DIVISIBLE_M:
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
            do = tl.load(do_ptrs)
            delta = tl.load(D + offs_m)
            l = tl.load(L + offs_m)
        else:
            mask_m = offs_m < M
            q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
            q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
            do = tl.load(do_ptrs, mask=mask_m[:, None])
            delta = tl.load(D + offs_m, mask=mask_m)
            l = tl.load(L + offs_m, mask=mask_m)
        piecewise_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :] + w
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(piecewise_mask, tl.dot(q2, k2), tl.dot(q1, k1))
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        if not DIVISIBLE_M:
            valid_mask = mask_m[:, None]
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            p = tl.where(causal_mask, p, 0.0)
        dv += tl.dot(tl.trans(p), do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_M:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds2 = tl.where(piecewise_mask, ds, 0.0)
        ds1 = tl.where(piecewise_mask, 0.0, ds)
        dk1 += tl.dot(tl.trans(ds1), q1)
        dk2 += tl.dot(tl.trans(ds2), q2)
        q1_ptrs += BLOCK_M * stride_q1m
        q2_ptrs += BLOCK_M * stride_q2m
        do_ptrs += BLOCK_M * stride_dom
    dk1 *= sm_scale
    dk2 *= sm_scale
    if DIVISIBLE_N:
        tl.store(dk1_ptrs, dk1)
        tl.store(dk2_ptrs, dk2)
        tl.store(dv_ptrs, dv)
    else:
        tl.store(dk1_ptrs, dk1, mask=mask_n[:, None])
        tl.store(dk2_ptrs, dk2, mask=mask_n[:, None])
        tl.store(dv_ptrs, dv, mask=mask_n[:, None])

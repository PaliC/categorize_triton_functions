import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kv_kernel(Q, K, V, B, sm_scale, DO, DK, DV, DS, L, D, stride_qz,
    stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
    stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_bz,
    stride_bh, stride_bm, stride_bn, stride_doz, stride_doh, stride_dom,
    stride_dok, stride_dkz, stride_dkh, stride_dkn, stride_dkk, stride_dvz,
    stride_dvh, stride_dvn, stride_dvk, Z, H, M, N, P_SEQ, lock, BLOCK_M:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    CAUSAL: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N:
    'tl.constexpr', HAS_BIAS: 'tl.constexpr', RETURN_DS: 'tl.constexpr',
    IS_BATCH_REDUCED: 'tl.constexpr', GROUP_SIZE_BIAS: 'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    if HAS_BIAS:
        if IS_BATCH_REDUCED:
            B += off_h * stride_bh
        else:
            B += off_z * stride_bz + off_h * stride_bh
    DO += off_z * stride_doz + off_h * stride_doh
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh
    if RETURN_DS:
        DS += off_z * stride_bz + off_h * stride_bh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = lo // BLOCK_M * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] *
        stride_qk)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] *
        stride_dok)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk
        )
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
        )
    if HAS_BIAS:
        bias_ptrs = B + (offs_m_init[:, None] * stride_bm + offs_n[None, :] *
            stride_bn)
    if RETURN_DS:
        ds_ptrs = DS + (offs_m_init[:, None] * stride_bm + offs_n[None, :] *
            stride_bn)
    mask_n = offs_n < N
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    else:
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        mask_m = offs_m < M
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            valid_mask = mask_m[:, None]
            q = tl.load(q_ptrs, mask=mask_m[:, None])
        if HAS_BIAS:
            if DIVISIBLE_M and DIVISIBLE_N:
                b = tl.load(bias_ptrs)
            else:
                b = tl.load(bias_ptrs, mask=mask_m[:, None] & mask_n[None, :])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_BIAS:
            s += b
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2((s - l[:, None]) * log2e)
        if not DIVISIBLE_M:
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)
        if DIVISIBLE_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None])
        dv += tl.dot(tl.trans(p), do)
        if DIVISIBLE_M:
            delta = tl.load(D + offs_m)
        else:
            delta = tl.load(D + offs_m, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_M:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds = ds
        dk += tl.dot(tl.trans(ds), q)
        if RETURN_DS:
            if DIVISIBLE_M and DIVISIBLE_N:
                tl.store(ds_ptrs, ds)
            else:
                tl.store(ds_ptrs, ds, mask=mask_m[:, None] & mask_n[None, :])
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if HAS_BIAS:
            bias_ptrs += BLOCK_M * stride_bm
        if RETURN_DS:
            ds_ptrs += BLOCK_M * stride_bm
    dk *= sm_scale
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk)
        tl.store(dv_ptrs, dv)
    else:
        tl.store(dk_ptrs, dk, mask=mask_n[:, None])
        tl.store(dv_ptrs, dv, mask=mask_n[:, None])

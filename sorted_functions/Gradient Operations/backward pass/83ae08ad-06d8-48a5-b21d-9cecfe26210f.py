import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kv_bias_kernel(Q, K, V, BW, sm_scale, DO, DK, DV, DB, L, D,
    cu_seqlens_q, cu_seqlens_k, nid_batch, nid_start, stride_qz, stride_qh,
    stride_qk, stride_kz, stride_kh, stride_kk, stride_vz, stride_vh,
    stride_vk, stride_doz, stride_doh, stride_dok, stride_dkz, stride_dkh,
    stride_dkk, stride_dvz, stride_dvh, stride_dvk, stride_bw, Z, H, M, N,
    BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', CAUSAL: 'tl.constexpr', HAS_BIAS: 'tl.constexpr',
    NUM_BUCKETS: 'tl.constexpr', MAX_DISTANCE: 'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
    log2e: 'tl.constexpr' = 1.4426950408889634
    start_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.load(nid_batch + start_z)
    off_n = tl.load(nid_start + start_z)
    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end = tl.load(cu_seqlens_k + off_b + 1)
    lM = q_end - q_start
    lN = k_end - k_start
    P_SEQ = lM - lN
    D += q_start * H + off_h
    L += q_start * H + off_h
    if CAUSAL:
        lo = tl.maximum(off_n - k_start - P_SEQ, 0)
        lo = lo // BLOCK_M * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M) + q_start
    offs_n = tl.arange(0, BLOCK_N) + off_n
    offs_n_relative = offs_n - k_start
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m_init[:, None] * stride_qz + offs_k[None, :] *
        stride_qk + off_h * stride_qh)
    k_ptrs = K + (offs_n[:, None] * stride_kz + offs_k[None, :] * stride_kk +
        off_h * stride_kh)
    v_ptrs = V + (offs_n[:, None] * stride_vz + offs_k[None, :] * stride_vk +
        off_h * stride_vh)
    do_ptrs = DO + (offs_m_init[:, None] * stride_doz + offs_k[None, :] *
        stride_dok + off_h * stride_doh)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvz + offs_k[None, :] *
        stride_dvk + off_h * stride_dvh)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkz + offs_k[None, :] *
        stride_dkk + off_h * stride_dkh)
    mask_n = offs_n < k_end
    v = tl.load(v_ptrs, mask=mask_n[:, None])
    k = tl.load(k_ptrs, mask=mask_n[:, None])
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    for start_m in range(lo, lM, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        if CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        mask_m = offs_m < lM
        valid_mask = mask_m[:, None]
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        if HAS_BIAS:
            relative_positions = offs_n_relative[None, :] - offs_m[:, None]
            relative_buckets = tl.zeros_like(relative_positions)
            num_buckets = NUM_BUCKETS
            if not CAUSAL:
                num_buckets //= 2
                relative_buckets += (relative_positions > 0) * num_buckets
                relative_positions = tl.abs(relative_positions)
            else:
                relative_positions = tl.maximum(-relative_positions, tl.
                    zeros_like(relative_positions))
            max_exact = num_buckets // 2
            is_small = relative_positions < max_exact
            relative_position_if_large = max_exact + tl.log(
                relative_positions.to(tl.float32) / max_exact) / tl.log(
                MAX_DISTANCE / max_exact) * (num_buckets - max_exact)
            relative_position_if_large = tl.minimum(relative_position_if_large,
                num_buckets - 1)
            relative_buckets += tl.where(is_small, relative_positions,
                relative_position_if_large)
            bucket_offs = relative_buckets * stride_bw + off_h
            bias_ptrs = BW + bucket_offs
            b = tl.load(bias_ptrs, mask=mask_m[:, None] & mask_n[None, :])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_BIAS:
            s += b
        l = tl.load(L + offs_m * H, mask=mask_m)
        p = tl.math.exp2((s - l[:, None]) * log2e)
        p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        dv += tl.dot(tl.trans(p), do)
        delta = tl.load(D + offs_m * H, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds = ds
        dk += tl.dot(tl.trans(ds), q)
        if HAS_BIAS:
            db_ptrs = DB + bucket_offs
            tl.atomic_add(db_ptrs, ds, mask=relative_buckets < NUM_BUCKETS)
        q_ptrs += BLOCK_M * stride_qz
        do_ptrs += BLOCK_M * stride_doz
    dk *= sm_scale
    tl.store(dk_ptrs, dk, mask=mask_n[:, None])
    tl.store(dv_ptrs, dv, mask=mask_n[:, None])

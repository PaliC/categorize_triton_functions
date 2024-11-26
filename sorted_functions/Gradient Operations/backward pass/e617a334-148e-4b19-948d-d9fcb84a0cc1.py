import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_q_kernel_with_bias_calculation(Q, K, V, BW, sm_scale, DO, DQ, L, D,
    cu_seqlens_q, cu_seqlens_k, mid_batch, mid_start, stride_qz, stride_qh,
    stride_qk, stride_kz, stride_kh, stride_kk, stride_vz, stride_vh,
    stride_vk, stride_doz, stride_doh, stride_dok, stride_dqz, stride_dqh,
    stride_dqk, stride_bw, Z, H, M, N, BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', CAUSAL:
    'tl.constexpr', HAS_BIAS: 'tl.constexpr', NUM_BUCKETS: 'tl.constexpr',
    MAX_DISTANCE: 'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
    log2e: 'tl.constexpr' = 1.4426950408889634
    start_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.load(mid_batch + start_z)
    off_m = tl.load(mid_start + start_z)
    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end = tl.load(cu_seqlens_k + off_b + 1)
    lM = q_end - q_start
    lN = k_end - k_start
    P_SEQ = lM - lN
    D += off_m * H + off_h
    L += off_m * H + off_h
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = offs_m_base + off_m
    offs_m_relative = offs_m - q_start
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    offs_n_init = k_start + offs_n_base
    q_ptrs = Q + (offs_m[:, None] * stride_qz + offs_k[None, :] * stride_qk +
        off_h * stride_qh)
    k_ptrs = K + (offs_n_init[:, None] * stride_kz + offs_k[None, :] *
        stride_kk + off_h * stride_kh)
    v_ptrs = V + (offs_n_init[:, None] * stride_vz + offs_k[None, :] *
        stride_vk + off_h * stride_vh)
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqz + offs_k[None, :] *
        stride_dqk + off_h * stride_dqh)
    do_ptrs = DO + (offs_m[:, None] * stride_doz + offs_k[None, :] *
        stride_dok + off_h * stride_doh)
    d_ptrs = D + offs_m_base * H
    l_ptrs = L + offs_m_base * H
    mask_m = offs_m < q_end
    q = tl.load(q_ptrs, mask=mask_m[:, None])
    do = tl.load(do_ptrs, mask=mask_m[:, None])
    delta = tl.load(d_ptrs, mask=mask_m)
    l = tl.load(l_ptrs, mask=mask_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if CAUSAL:
        hi = tl.minimum(lN, P_SEQ + off_m - q_start + BLOCK_M)
        if lM > lN:
            hi = tl.maximum(0, hi)
    else:
        hi = lN
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        mask_n = offs_n < lN
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])
        if HAS_BIAS:
            relative_positions = offs_n[None, :] - offs_m_relative[:, None]
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
        if CAUSAL:
            causal_mask = P_SEQ + offs_m_relative[:, None] >= offs_n[None, :]
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_BIAS:
            s += b
        p = tl.math.exp2((s - l[:, None]) * log2e)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        ds = tl.where(mask_n, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        dq += tl.dot(ds, k)
        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz
    dq *= sm_scale
    tl.store(dq_ptrs, dq, mask=mask_m[:, None])

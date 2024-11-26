import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_with_bias_calculation(Q, K, V, BW, sm_scale, L, O,
    cu_seqlens_q, cu_seqlens_k, mid_batch, mid_start, stride_qz, stride_qh,
    stride_qk, stride_kz, stride_kh, stride_kk, stride_vz, stride_vh,
    stride_vk, stride_oz, stride_oh, stride_ok, stride_wn, Z, H, M, N,
    BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', IS_CAUSAL: 'tl.constexpr', HAS_BIAS: 'tl.constexpr',
    NUM_BUCKETS: 'tl.constexpr', MAX_DISTANCE: 'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
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
    log2e: 'tl.constexpr' = 1.4426950408889634
    L += off_m * H + off_h
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = offs_m_base + off_m
    offs_m_relative = offs_m - q_start
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qz + off_h * stride_qh + offs_k[
        None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_oz + off_h * stride_oh + offs_k[
        None, :] * stride_ok)
    l_ptrs = L + offs_m_base * H
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    mask_m = offs_m < q_end
    q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier='.cg')
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k, tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 1.0, dtype=input_dtype), tl.full((BLOCK_DMODEL,
            BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I)
    if IS_CAUSAL:
        hi = tl.minimum(lN, P_SEQ + (off_m + 1) * BLOCK_M)
        if lM > lN:
            hi = tl.maximum(0, hi)
    else:
        hi = lN
    offs_n_init = k_start + offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] *
        stride_vz + off_h * stride_kh)
    v_ptrs = V + (offs_n_init[:, None] * stride_kz + offs_k[None, :] *
        stride_kk + off_h * stride_vh)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        mask_n = offs_n < lN
        k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier='.cg')
        v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier='.cg')
        if HAS_BIAS:
            relative_positions = offs_n[None, :] - offs_m_relative[:, None]
            relative_buckets = tl.zeros_like(relative_positions)
            num_buckets = NUM_BUCKETS
            if not IS_CAUSAL:
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
            bucket_offs = relative_buckets * stride_wn + off_h
            bias_ptrs = BW + bucket_offs
            bias_values = tl.load(bias_ptrs, mask_m[:, None] & mask_n[None, :])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k) * sm_scale
        if HAS_BIAS:
            s += bias_values
        s = tl.where(mask_n[None, :], s, float('-inf'))
        if IS_CAUSAL:
            causal_mask = P_SEQ + offs_m_relative[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        p = tl.math.exp2((s - m_i_new[:, None]) * log2e)
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz
    if IS_CAUSAL and lM > lN:
        is_empty_line = offs_m_relative + P_SEQ < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float('-inf'), m_i + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i + tl.log(l_i)
    tl.store(l_ptrs, l, mask=mask_m, cache_modifier='.cg')
    tl.store(o_ptrs, acc, mask=mask_m[:, None], cache_modifier='.cg')

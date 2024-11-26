import triton
import triton.language as tl
import torch

@triton.jit
def attn_fwd(Q, K, V, B, sm_scale, L, Out, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
    stride_vh, stride_vk, stride_vn, stride_bz, stride_bh, stride_bm,
    stride_bn, stride_oz, stride_oh, stride_om, stride_on, num_head_q:
    "'i32'", num_head_k: "'i32'", cu_seqlens_q, cu_seqlens_k, num_seqlens:
    "'i32'", max_seqlen_q: "'i32'", max_seqlen_k: "'i32'", head_dim:
    "'i32'", dropout_p, philox_seed_ptr, philox_offset1: "'*u32'",
    philox_offset2: "'i32'", philox_seed_output: "'*u64'",
    philox_offset_output: "'*u64'", encoded_softmax, CAUSAL: 'tl.constexpr',
    BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', pre_load_v: 'tl.constexpr', ENABLE_DROPOUT:
    'tl.constexpr', RETURN_ENCODED_SOFTMAX: 'tl.constexpr', PADDED_HEAD:
    'tl.constexpr', BIAS_TYPE: 'tl.constexpr'):
    PRE_LOAD_V: 'tl.constexpr' = pre_load_v
    USE_ALIBI: 'tl.constexpr' = False
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
        if (tl.program_id(0) == 0 and tl.program_id(1) == 0) and tl.program_id(
            2) == 0:
            if philox_seed_output.cast(dtype=tl.uint64, bitcast=True) != 0:
                tl.store(philox_seed_output, philox_seed)
            if philox_offset_output.cast(dtype=tl.uint64, bitcast=True) != 0:
                tl.store(philox_offset_output, philox_offset_base)
    if num_seqlens > 0:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        batch_index = 0
    elif num_seqlens == 0:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
        batch_index = off_z
    else:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if CAUSAL:
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M, BLOCK_N)
        n_blocks = min(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            o_offset = (Out + batch_index * stride_oz + off_h_q * stride_oh +
                cu_seqlens_q_start * stride_om)
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :
                ] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            L_ptr_base = L + (off_z * num_head_q + off_h_q) * max_seqlen_q
            l_ptrs = L_ptr_base + offs_m
            l = tl.full([BLOCK_M], value=float('inf'), dtype=tl.float32)
            l_ptrs_mask = offs_m < max_seqlen_q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            return
    off_h_k = off_h_q // (num_head_q // num_head_k)
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    q_offset = (Q + batch_index * stride_qz + off_h_q * stride_qh + 
        cu_seqlens_q_start * stride_qm)
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :
        ] * stride_qk
    k_offset = (batch_index * stride_kz + off_h_k * stride_kh + 
        cu_seqlens_k_start * stride_kn)
    k_ptrs = K + k_offset + offs_d[:, None] * stride_kk + offs_n[None, :
        ] * stride_kn
    v_offset = (V + batch_index * stride_vz + off_h_k * stride_vh + 
        cu_seqlens_k_start * stride_vk)
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :
        ] * stride_vn
    if BIAS_TYPE == 0:
        bias_ptrs = None
    elif BIAS_TYPE == 1:
        bias_offset = off_h_q * stride_bh
        bias_ptrs = B + bias_offset + offs_m[:, None] * stride_bm + offs_n[
            None, :] * stride_bn
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    if USE_ALIBI:
        a_offset = batch_index * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None
    off_zh = batch_index * num_head_q + off_h_q
    if ENABLE_DROPOUT:
        batch_philox_offset = (philox_offset_base + off_zh * max_seqlen_q *
            max_seqlen_k)
    else:
        batch_philox_offset = 0
    if RETURN_ENCODED_SOFTMAX:
        encoded_sm_base = (encoded_softmax + off_zh * max_seqlen_q *
            max_seqlen_k)
    else:
        encoded_sm_base = None
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and seqlen_q % BLOCK_M == 0
    if CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = attn_fwd_inner(acc, l_i, m_i, qk_scale, bias_scale,
            q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn,
            seqlen_q, seqlen_k, head_dim, start_m, block_min, block_max,
            dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
            encoded_sm_base, 0, 0, 0, alibi_slope, False, BLOCK_M,
            BLOCK_DMODEL, BLOCK_N, PRE_LOAD_V, False, ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX, PADDED_HEAD)
        block_min = block_max
        block_max = n_blocks * BLOCK_N
    tl.debug_barrier()
    if masked_blocks > 0:
        if CAUSAL:
            offs_n_causal = offs_n
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        acc, l_i, m_i = attn_fwd_inner(acc, l_i, m_i, qk_scale, bias_scale,
            q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn,
            seqlen_q, seqlen_k, head_dim, start_m, block_min, block_max,
            dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
            encoded_sm_base, offs_n_causal, masked_blocks, n_extra_tokens,
            alibi_slope, CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, PRE_LOAD_V,
            True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD)
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = 0
    acc = acc
    if CAUSAL:
        tl.assume(start_m_idx >= 0)
        tl.assume(end_m_idx >= 0)
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL,), causal_start_idx,
                dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[
                None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z)
    L_ptr_base = L + (off_z * num_head_q + off_h_q) * max_seqlen_q
    l_ptrs = L_ptr_base + offs_m
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    o_base = (Out + batch_index * stride_oz + off_h_q * stride_oh + 
        cu_seqlens_q_start * stride_om)
    mstore2d(acc, BLOCK_M, BLOCK_DMODEL, o_base=o_base, o_start_row=start_m *
        BLOCK_M, o_start_col=0, o_rows=seqlen_q, o_cols=head_dim,
        stride_row=stride_om, stride_col=stride_on)

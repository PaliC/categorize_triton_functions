import triton
import triton.language as tl
import torch

@triton.jit
def attn_fwd_inner(acc, l_i, m_i, qk_scale, bias_scale, q, k_ptrs, v_ptrs,
    bias_ptrs, stride_kn, stride_vk, stride_bn, seqlen_q, seqlen_k,
    head_dim, start_m, block_min, block_max, dropout_p, philox_seed,
    batch_philox_offset, max_seqlen_k, encoded_sm_base, offs_n_causal,
    masked_blocks, n_extra_tokens, alibi_slope, CAUSAL: 'tl.constexpr',
    BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', PRE_LOAD_V: 'tl.constexpr', MASK_STEPS: 'tl.constexpr',
    ENABLE_DROPOUT: 'tl.constexpr', RETURN_ENCODED_SOFTMAX: 'tl.constexpr',
    PADDED_HEAD: 'tl.constexpr'):
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(block_min, block_max, BLOCK_N):
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_d, k_offs_n, head_dim, seqlen_k)
        if PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_d, seqlen_k, head_dim)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MASK_STEPS:
            if start_n + BLOCK_N == block_max and n_extra_tokens != 0:
                boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
                size_n = start_n + offs_n[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float('-inf'))
        if CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = offs_m[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N
                ) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, offs_m, bias_offs_n, seqlen_q, seqlen_k)
            qk += bias * bias_scale
        if alibi_slope is not None:
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, seqlen_q,
                seqlen_k, global_m_positions, global_n_positions)
            qk += alibi_block * bias_scale
        m_ij = tl.maximum(m_i, qk_scale * tl.max(qk, 1))
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        if MASK_STEPS or CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = (batch_philox_offset + start_m * BLOCK_M *
                max_seqlen_k + start_n)
            keep = dropout_mask(philox_seed, philox_offset, dropout_p,
                BLOCK_M, BLOCK_N, max_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                mstore2d(tl.where(keep, p, -p), BLOCK_M, BLOCK_N, o_base=
                    encoded_sm_base, o_start_row=start_m * BLOCK_M,
                    o_start_col=start_n, o_rows=seqlen_q, o_cols=seqlen_k,
                    stride_row=max_seqlen_k, stride_col=1)
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            mstore2d(p, BLOCK_M, BLOCK_N, o_base=encoded_sm_base,
                o_start_row=start_m * BLOCK_M, o_start_col=start_n, o_rows=
                seqlen_q, o_cols=seqlen_k, stride_row=max_seqlen_k,
                stride_col=1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_d, seqlen_k, head_dim)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc += tl.dot(p, v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
    return acc, l_i, m_i

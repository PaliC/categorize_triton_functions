import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m,
    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed,
    batch_philox_offset, encoded_softmax_block_ptr, block_min, block_max,
    offs_n_causal, masked_blocks, n_extra_tokens, bias_ptr, alibi_slope,
    IS_CAUSAL: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', OFFS_M: 'tl.constexpr', OFFS_N:
    'tl.constexpr', PRE_LOAD_V: 'tl.constexpr', MASK_STEPS: 'tl.constexpr',
    ENABLE_DROPOUT: 'tl.constexpr', RETURN_ENCODED_SOFTMAX: 'tl.constexpr',
    PADDED_HEAD: 'tl.constexpr'):
    for start_n in range(block_min, block_max, BLOCK_N):
        k = load_fn(K_block_ptr, PADDED_HEAD, MASK_STEPS and n_extra_tokens !=
            0, 'zero')
        if PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and n_extra_tokens != 0,
                PADDED_HEAD, 'zero')
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MASK_STEPS:
            if start_n + BLOCK_N == block_max and n_extra_tokens != 0:
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32
                    )
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float('-inf'))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        if bias_ptr is not None:
            bias = load_fn(bias_ptr, False, MASK_STEPS and n_extra_tokens !=
                0, 'zero')
            qk += bias * 1.44269504089
        if alibi_slope is not None:
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            relative_pos_block = global_m_positions[:, None
                ] + actual_seqlen_k - global_n_positions[None, :
                ] - actual_seqlen_q
            relative_pos_block = tl.abs(relative_pos_block)
            alibi_block = -1 * alibi_slope * relative_pos_block
            qk += alibi_block * 1.44269504089
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = (batch_philox_offset + start_m * BLOCK_M *
                actual_seqlen_k + start_n - BLOCK_N)
            keep = dropout_mask(philox_seed, philox_offset, dropout_p,
                BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_softmax_block_ptr, tl.where(keep, p, -p))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr, p)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and n_extra_tokens != 0,
                PADDED_HEAD, 'zero')
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc += tl.dot(p, v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr,
                (0, BLOCK_N))
    return acc, l_i, m_i

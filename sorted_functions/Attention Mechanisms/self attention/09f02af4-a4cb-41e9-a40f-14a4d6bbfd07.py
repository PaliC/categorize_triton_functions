import triton
import triton.language as tl
import torch

@triton.heuristics({'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None}
    )
@triton.jit
def parallel_delta_rule_fwd_kernel(q, k, k2, v, beta, o, o_new, attn, s_k_h,
    s_k_t, s_v_h, s_v_t, T: 'tl.constexpr', K: 'tl.constexpr', V:
    'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', OUTPUT_ATTENTIONS: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT,
        0), (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT,
        0), (BT, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))
    for offset in range((i_t + 1) * BT - 2 * BS, i_t * BT - BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (0,
            offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (
            offset, 0), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (
            offset, 0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (offset,),
            (BS,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        m_s = tl.arange(0, BT) >= offset - i_t * BT + BS
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s[:, None], b_s, 0)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        b_k2 = tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]
        b_q -= tl.dot(b_s, b_k2, allow_tf32=False)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (
                i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s, boundary_check=(0, 1))
    for offset in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (0,
            offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (
            offset, 0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (offset,),
            (BS,), (0,))
        p_k2 = tl.make_block_ptr(k2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (
            offset, 0), (BS, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        b_k2 = tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (
                i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s, boundary_check=(0, 1))
    p_o_new = tl.make_block_ptr(o_new + i_bh * s_v_h, (T, V), (s_v_t, 1), (
        i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o, boundary_check=(0, 1))

import triton
import triton.language as tl
import torch

@triton.heuristics({'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None})
@triton.jit
def parallel_simple_gla_fwd_kernel(q, k, v, g, o, attn, s_k_h, s_k_t, s_v_h,
    s_v_t, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr',
    K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NV:
    'tl.constexpr', OUTPUT_ATTENTIONS: 'tl.constexpr'):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT,
        i_k * BK), (BT, BK), (1, 0))
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k * B * H + i_bh) * T * T, (T, T),
            (T, 1), (i_t * BT, 0), (BT, BS), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k *
            BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s,
            i_v * BV), (BS, BV), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_gn = tl.load(g + i_bh * T + min(i_s + BS, T) - 1)
        b_gp = tl.load(g + i_bh * T + i_s - 1) if i_s % BT > 0 else 0.0
        b_kg = b_k * tl.exp(b_gn - b_g)
        b_s = tl.dot(b_q, b_kg, allow_tf32=False)
        if i_s > 0:
            b_o = b_o * tl.exp(b_gn - b_gp)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s, boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))
    tl.debug_barrier()
    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_gq = tl.load(p_g, boundary_check=(0,))
    b_o *= tl.exp(b_gq)[:, None]
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k * B * H + i_bh) * T * T, (T, T),
            (T, 1), (i_t * BT, i_t * BT), (BT, BS), (1, 0))
    o_q = i_t * BT + tl.arange(0, BT)
    o_k = i_t * BT + tl.arange(0, BS)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k *
            BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s,
            i_v * BV), (BS, BV), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0,))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.where(m_s, tl.dot(b_q, b_k, allow_tf32=False) * tl.exp(
            b_gq[:, None] - b_gk[None, :]), 0)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s, boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))
        o_k += BS
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_v_h, (T, V), (
        s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))

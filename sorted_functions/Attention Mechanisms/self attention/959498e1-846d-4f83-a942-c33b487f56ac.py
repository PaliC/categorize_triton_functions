import triton
import triton.language as tl
import torch

@triton.heuristics({'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None})
@triton.jit
def parallel_retention_fwd_kernel(q, k, v, o, attn, scale, B:
    'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr',
    V: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NV: 'tl.constexpr',
    OUTPUT_ATTENTIONS: 'tl.constexpr'):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_k = tl.arange(0, BS)
    d_h = tl.math.exp2((BS - o_k) * b_b)
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (i_k * BK, 0),
        (BK, BS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (0, i_v * BV),
        (BS, BV), (1, 0))
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k * B * H + i_bh) * T * T, (T, T),
            (T, 1), (i_t * BT, 0), (BT, BS), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i in range(0, i_t * BT, BS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_h
        if i > 0:
            b_o = b_o * tl.math.exp2(b_b * BS)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BS))
        p_v = tl.advance(p_v, (BS, 0))
        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s, boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))
    tl.debug_barrier()
    o_q = tl.arange(0, BT)
    d_q = tl.math.exp2(tl.arange(0, BT) * b_b)
    b_o *= d_q[:, None]
    p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (i_k * BK, 
        i_t * BT), (BK, BS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 
        i_v * BV), (BS, BV), (1, 0))
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k * B * H + i_bh) * T * T, (T, T),
            (T, 1), (i_t * BT, i_t * BT), (BT, BS), (1, 0))
    for _ in range(i_t * BT, (i_t + 1) * BT, BS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) *
            b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s, boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))
        p_k = tl.advance(p_k, (0, BS))
        p_v = tl.advance(p_v, (BS, 0))
        o_k += BS
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * T * V, (T, V), (V, 1
        ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))

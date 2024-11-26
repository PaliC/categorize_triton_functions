import triton
import triton.language as tl
import torch

@triton.jit
def fused_chunk_gla_fwd_kernel(q, k, v, g, o, h0, ht, s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d, B: 'tl.constexpr', H: 'tl.constexpr', T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr',
    CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (0, 
        i_k * BK), (BT, BK), (1, 0))
    p_db = g + i_bh * s_k_h + (BT - 1) * s_k_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k *
        BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (0, 
        i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_v_h, (T, V), (
        s_v_t, s_v_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    mask = i_k * BK + tl.arange(0, BK) < K
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        d_b = tl.load(p_db, mask=mask, other=0)
        if CHECK and i == 0:
            b_o = tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h * tl.exp(d_b)[:, None] + tl.dot(b_k, b_v, allow_tf32=
                False)
        else:
            b_o = tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h * tl.exp(d_b)[:, None] + tl.dot(b_k, b_v, allow_tf32=
                False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_db += BT * K
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h, boundary_check=(0, 1))

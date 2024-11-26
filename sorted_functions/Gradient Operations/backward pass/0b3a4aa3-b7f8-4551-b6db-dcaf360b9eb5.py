import triton
import triton.language as tl
import torch

@triton.jit
def chunk_rwkv6_bwd_kernel_dh(q, g, gs, do, dh, dh0, s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        o_t = min(i_t * BT + BT, T)
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
            (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            o_t - 1) * K + i_k * BK,), (BK,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        b_gn = tl.load(p_gn, boundary_check=(0,))
        b_dh *= tl.exp(b_gn)[:, None]
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        b_q = b_q * tl.exp(b_gs)
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)
    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh, boundary_check=(0, 1))

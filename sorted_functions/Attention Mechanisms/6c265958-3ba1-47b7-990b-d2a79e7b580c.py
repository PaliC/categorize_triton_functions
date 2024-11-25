import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK', 'BV', 'USE_G', 'USE_GK', 'USE_GV'])
@triton.heuristics({'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'
    ] is not None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not
    None})
@triton.jit
def chunk_bwd_kernel_dh(q, g, gk, gv, do, dh, dht, dh0, s_k_h, s_k_t, s_v_h,
    s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V:
    'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NT: 'tl.constexpr', NG: 'tl.constexpr', USE_G:
    'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr',
    STORE_INITIAL_STATE_GRADIENT: 'tl.constexpr', USE_FINAL_STATE_GRADIENT:
    'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k *
            BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (
            i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        if USE_G:
            p_g = g + i_bg * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
            b_g = tl.load(p_g, mask=i_t * BT + tl.arange(0, BT) < T, other=0.0)
            b_q = b_q * tl.exp(b_g)[None, :]
            b_g_last = tl.load(g + i_bg * T + last_idx)
            b_dh *= tl.exp(b_g_last)
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + i_bg * s_k_h, (K, T), (1, s_k_t),
                (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_q = b_q * tl.exp(b_gk)
            p_gk_last = (gk + i_bg * s_k_h + last_idx * K + i_k * BK + tl.
                arange(0, BK))
            p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
            b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) <
                K, other=0.0)
            b_dh *= tl.exp(b_gk_last)[:, None]
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + i_bg * s_v_h, (T, V), (s_v_t, 1),
                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_do = b_do * tl.exp(b_gv)
            p_gv_last = (gv + i_bg * s_v_h + last_idx * V + i_v * BV + tl.
                arange(0, BV))
            p_gv_last = tl.max_contiguous(tl.multiple_of(p_gv_last, BV), BV)
            b_gv_last = tl.load(p_gv_last, mask=i_v * BV + tl.arange(0, BV) <
                V, other=0.0)
            b_dh *= tl.exp(b_gv_last)[None, :]
        b_dh += tl.dot(b_q, b_do)
    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh, boundary_check=(0, 1))

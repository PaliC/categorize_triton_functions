import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK', 'BV', 'USE_G', 'USE_GK', 'USE_GV'])
@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None})
@triton.jit
def chunk_fwd_kernel_h(k, v, h, g, gk, gv, h0, ht, s_k_h, s_k_t, s_v_h,
    s_v_t, s_h_h, s_h_t, T: 'tl.constexpr', K: 'tl.constexpr', V:
    'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NT: 'tl.constexpr', USE_G: 'tl.constexpr', USE_GK:
    'tl.constexpr', USE_GV: 'tl.constexpr', USE_INITIAL_STATE:
    'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k *
            BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + i_bh * T + last_idx)
            b_h *= tl.exp(b_g_last)
            p_g = g + i_bh * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
            b_g = tl.load(p_g, mask=i_t * BT + tl.arange(0, BT) < T, other=0.0)
            b_v = b_v * tl.exp(b_g_last - b_g)[:, None]
        if USE_GK:
            p_gk_last = (gk + i_bh * s_k_h + last_idx * K + i_k * BK + tl.
                arange(0, BK))
            p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
            b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) <
                K, other=0.0)
            b_h *= tl.exp(b_gk_last)[:, None]
            p_gk = tl.make_block_ptr(gk + i_bh * s_k_h, (K, T), (1, s_k_t),
                (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = b_k * tl.exp(b_gk_last[:, None] - b_gk)
        if USE_GV:
            p_gv_last = (gv + i_bh * s_v_h + last_idx * V + i_v * BV + tl.
                arange(0, BV))
            p_gv_last = tl.max_contiguous(tl.multiple_of(p_gv_last, BV), BV)
            b_gv_last = tl.load(p_gv_last, mask=i_v * BV + tl.arange(0, BV) <
                V, other=0.0)
            b_h *= tl.exp(b_gv_last)[None, :]
            p_gv = tl.make_block_ptr(gv + i_bh * s_v_h, (T, V), (s_v_t, 1),
                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = b_v * tl.exp(b_gv_last[None, :] - b_gv)
        b_h += tl.dot(b_k, b_v)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))

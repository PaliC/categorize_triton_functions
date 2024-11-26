import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_h(k, v, g, h, h0, ht, s_k_h, s_k_t, s_k_d, s_v_h,
    s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, T: 'tl.constexpr', K: 'tl.constexpr',
    V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NT: 'tl.constexpr', NORMK: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if NORMK:
            p_g = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t
                ), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,),
                ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
            b_gn = tl.load(p_gn, boundary_check=(0,))
            b_h *= tl.exp(b_gn)[:, None]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_k = b_k * tl.exp(b_gn[:, None] - b_g)
        else:
            p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d
                ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,),
                ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
            b_gn = tl.load(p_gn, boundary_check=(0,))
            b_h *= tl.exp(b_gn)[None, :]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_v = b_v * tl.exp(b_gn[None, :] - b_g)
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))

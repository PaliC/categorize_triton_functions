import triton
import triton.language as tl
import torch

@triton.jit
def chunk_linear_attn_fwd_kernel_h(k, v, h, h0, ht, s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))
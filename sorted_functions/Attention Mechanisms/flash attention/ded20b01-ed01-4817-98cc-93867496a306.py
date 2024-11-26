import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gla_fwd_kernel(k, v, g, h, initial_state, final_state, s_qk_h,
    s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_hh, s_ht, B, H, T, scale, BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK:
    'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (
        i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (
        0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (
            DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        d_b = tl.load(p_db)
        p_h = tl.make_block_ptr(h + i_bh * s_hh, ((i + 1) * DK, DV), (s_ht,
            1), (i * DK + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_h *= tl.math.exp2(d_b)[:, None]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_db += BT * DK
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV),
            (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h, boundary_check=(0, 1))
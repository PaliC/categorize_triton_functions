import triton
import triton.language as tl
import torch

@triton.jit
def chunk_simple_gla_bwd_kernel_dh(q, g, do, dh, s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H: 'tl.constexpr', T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'
    ):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t,
            s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale * tl.math.exp2(tl.load(g + i_bh * T + i_t * BT +
            tl.arange(0, BT)))[None, :]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh *= tl.math.exp2(tl.load(g + i_bh * T + i_t * BT + BT - 1))
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)

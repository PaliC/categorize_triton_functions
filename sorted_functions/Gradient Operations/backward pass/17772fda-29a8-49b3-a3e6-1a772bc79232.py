import triton
import triton.language as tl
import torch

@triton.jit
def chunk_delta_rule_bwd_kernel_dhv(q, k, d, do, dh, dv, s_qk_h, s_qk_t,
    s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H: 'tl.constexpr',
    T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'
    ):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    m_s = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t,
            s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t,
            s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dv = tl.dot(b_s, b_do, allow_tf32=False) + tl.dot(b_k, b_dh,
            allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        b_dh -= tl.dot(b_d, b_dv, allow_tf32=False)

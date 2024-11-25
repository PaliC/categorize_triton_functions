import triton
import triton.language as tl
import torch

@triton.jit
def chunk_retention_fwd_kernel_o(q, k, v, h, o, s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d, s_hh, s_ht, H, T, TD, scale, DK, DV, BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    for i_v in range(0, tl.cdiv(DV, BV)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (i_t * BT, 0), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t
            ), (0, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (TD, DV), (s_ht, 1), (i_t *
            DK, i_v * BV), (BK, BV), (1, 0))
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_s = tl.zeros([BT, BT], dtype=tl.float32)
        for _ in range(0, tl.cdiv(DK, BK)):
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = b_q * scale
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q * d_i[:, None], b_h, allow_tf32=False)
            b_s += tl.dot(b_q, b_k, allow_tf32=False)
            p_q = tl.advance(p_q, (0, BK))
            p_k = tl.advance(p_k, (BK, 0))
            p_h = tl.advance(p_h, (BK, 0))
        b_s *= d_s
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d
            ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d
            ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o, boundary_check=(0, 1))

import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gla_bwd_kernel(q, g, do, dh, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
    s_vo_t, s_vo_d, s_hh, s_ht, B, H, T, TDK, scale, BT: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'
    ):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)[:, None] < DK) & (i_v * BV + tl.
        arange(0, BV)[None, :] < DV)
    p_dh = dh + i_bh * s_hh + (TDK - DK + i_k * BK + tl.arange(0, BK)[:, None]
        ) * DV + i_v * BV + tl.arange(0, BV)[None, :]
    for i in range((tl.cdiv(T, BT) - 1) * BT, -BT, -BT):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t
            ), (i_k * BK, i), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t,
            s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_db = g + i_bh * s_qk_h + (i + BT - 1
            ) * s_qk_t + i_k * BK + tl.arange(0, BK)
        d_b = tl.math.exp2(tl.load(p_db))
        tl.store(p_dh, b_dh, mask=mask)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = d_b[:, None] * b_dh + tl.dot(b_q, b_do, allow_tf32=False)
        p_dh -= DK * DV

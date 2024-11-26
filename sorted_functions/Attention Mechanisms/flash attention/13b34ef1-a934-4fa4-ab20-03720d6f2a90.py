import triton
import triton.language as tl
import torch

@triton.jit
def chunk_retention_bwd_kernel_dqkv(q, k, v, h, do, dh, dq, dk, dv, s_qk_h,
    s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_hh, s_ht, H, T, TDK, scale,
    DK, DV, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b
        )
    d_q = d_q * scale
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0
        ) * scale
    for i_k in range(0, tl.cdiv(DK, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t
            ), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d
            ), (i_t * BT, 0), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (DV, TDK), (1, s_ht), (0, 
            i_t * DK + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t,
            s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_hh, (TDK, DV), (s_ht, 1), (
            i_t * DK + i_k * BK, 0), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), (s_vo_t,
            s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, DK), (s_qk_t,
            s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, DK), (s_qk_t,
            s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        for _ in range(tl.cdiv(DV, BV)):
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_ds = tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
            b_ds = b_ds * d_s
            b_dq += tl.dot(b_do, b_h, allow_tf32=False) * d_q[:, None
                ] + tl.dot(b_ds, b_k, allow_tf32=False)
            b_ds = tl.trans(b_ds)
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * d_k[:, None
                ]
            b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
            b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(
                b_s, b_do, allow_tf32=False)
            b_dv += tl.load(p_dv, boundary_check=(0, 1))
            tl.store(p_dv, b_dv, boundary_check=(0, 1))
            p_v = tl.advance(p_v, (0, BV))
            p_h = tl.advance(p_h, (BV, 0))
            p_do = tl.advance(p_do, (0, BV))
            p_dh = tl.advance(p_dh, (0, BV))
            p_dv = tl.advance(p_dv, (0, BV))
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
        tl.store(p_dk, b_dk, boundary_check=(0, 1))

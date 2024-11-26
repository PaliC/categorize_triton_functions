import triton
import triton.language as tl
import torch

@triton.jit
def parallel_simple_gla_bwd_kernel_dq(i_bh, i_t, i_k, i_v, q, k, v, g, do,
    dq, dg, s_k_h, s_k_t, s_v_h, s_v_t, scale, B: 'tl.constexpr', H:
    'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BS: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr'):
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t *
        BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s,
            i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v *
            BV, i_s), (BV, BS), (0, 1))
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_gn = tl.load(g + i_bh * T + min(i_s + BS, T) - 1)
        b_gp = tl.load(g + i_bh * T + i_s - 1) if i_s % BT > 0 else 0.0
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * tl.exp(b_gn - b_g)[None, :
            ]
        if i_s > 0:
            b_dq *= tl.exp(b_gn - b_gp)
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    p_gq = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    b_gq = tl.load(p_gq, boundary_check=(0,))
    b_dq *= tl.exp(b_gq)[:, None] * scale
    o_q = i_t * BT + tl.arange(0, BT)
    o_k = i_t * BT + tl.arange(0, BS)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s,
            i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v *
            BV, i_s), (BV, BS), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0,))
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.where(m_s, tl.dot(b_do, b_v, allow_tf32=False) * tl.exp(
            b_gq[:, None] - b_gk[None, :]), 0) * scale
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        o_k += BS
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT,
        i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (i_v * B * H + i_bh) * s_k_h, (T, K), (
        s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_v * B * H + i_bh) * T, (T,), (1,), (
        i_t * BT,), (BT,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_dg = tl.sum(b_dq * b_q, 1)
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0,))

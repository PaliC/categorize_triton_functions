import triton
import triton.language as tl
import torch

@triton.jit
def parallel_simple_gla_bwd_kernel_dkv(i_bh, i_t, i_k, i_v, q, k, v, g, do,
    dk, dv, dg, s_k_h, s_k_t, s_v_h, s_v_t, scale, B: 'tl.constexpr', H:
    'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BS: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr'):
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT,
        i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT,
        i_v * BV), (BT, BV), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    b_gk = tl.load(p_gk, boundary_check=(0,))
    NTS = tl.cdiv(T, BS)
    b_kg = b_k * tl.exp(tl.load(g + i_bh * T + min(i_t * BT + BT, T) - 1) -
        b_gk)[:, None]
    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s,
            i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (
            i_s, i_v * BV), (BS, BV), (1, 0))
        p_gq = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.load(p_gq, boundary_check=(0,))
        b_gp = tl.load(g + i_bh * T + min(i_s + BS, T) - 1)
        b_gn = tl.load(g + i_bh * T + i_s - 1) if i_s % BT > 0 else 0.0
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_gq - b_gn)[:, None]
        b_dk *= tl.exp(b_gp - b_gn)
        b_dv *= tl.exp(b_gp - b_gn)
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_s = tl.dot(b_kg, tl.trans(b_q), allow_tf32=False)
        b_dk += tl.dot(b_ds, b_q, allow_tf32=False)
        b_dv += tl.dot(b_s, b_do, allow_tf32=False)
    b_dk *= tl.exp(tl.load(g + i_bh * T + min(T, i_t * BT + BT) - 1) - b_gk)[
        :, None] * scale
    b_dv *= scale
    tl.debug_barrier()
    o_q = i_t * BT + tl.arange(0, BS)
    o_k = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s,
            i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (
            i_s, i_v * BV), (BS, BV), (1, 0))
        p_gq = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_gq = tl.load(p_gq, boundary_check=(0,))
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.exp(-b_gk[:, None] + b_gq[None, :]), 0) * scale
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False) * d_s
        b_s = tl.dot(b_k, tl.trans(b_q), allow_tf32=False) * d_s
        b_dk += tl.dot(b_ds, b_q, allow_tf32=False)
        b_dv += tl.dot(b_s, b_do, allow_tf32=False)
        o_q += BS
    p_dk = tl.make_block_ptr(dk + (i_v * B * H + i_bh) * s_k_h, (T, K), (
        s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_k * B * H + i_bh) * s_v_h, (T, V), (
        s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_v * B * H + i_bh) * T, (T,), (1,), (
        i_t * BT,), (BT,), (0,))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    b_dg = tl.load(p_dg, boundary_check=(0,))
    b_dg -= tl.sum(b_dk * b_k, 1)
    tl.store(p_dg, b_dg, boundary_check=(0,))

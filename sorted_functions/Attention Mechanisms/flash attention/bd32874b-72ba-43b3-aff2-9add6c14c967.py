import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(q, k, v, h, g, do, dh, dq, dk, dg,
    scale, B: 'tl.constexpr', T: 'tl.constexpr', H: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', HEAD_FIRST:
    'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)
    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,
            ), (0,))
        b_g_last = tl.load(g + i_bh * T + min(i_t * BT + BT, T) - 1)
    else:
        p_g = tl.make_block_ptr(g + i_b * T * H + i_h, (T,), (H,), (i_t *
            BT,), (BT,), (0,))
        b_g_last = tl.load(g + i_b * T * H + (min(i_t * BT + BT, T) - 1) * H)
    b_g = tl.load(p_g, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t *
                BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (
                i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V, (V, NT * K), (1, V),
            (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * NT * K * V, (V, NT * K), (1, V
            ), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, b_dh)
    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT,
            i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT,
            i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + (i_k * B * H + i_bh) * T, (T,), (1,),
            (i_t * BT,), (BT,), (0,))
    else:
        p_q = tl.make_block_ptr(q + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BK, BT), (0, 1))
        p_dq = tl.make_block_ptr(dq + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BK, BT), (0, 1))
        p_dk = tl.make_block_ptr(dk + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BK, BT), (0, 1))
        p_dg = tl.make_block_ptr(dg + (i_k * B + i_b) * T * H + i_h, (T,),
            (H,), (i_t * BT,), (BT,), (0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_dg_last *= tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(b_g
        [:, None] - b_g[None, :]), 0)
    b_ds = b_ds
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    b_dg = tl.where(o_i < min(BT, T - i_t * BT) - 1, b_dg, b_dg + b_dg_last)
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0,))

import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_simple_gla_bwd_kernel_dv(q, k, g, do, dv, dh, scale, T:
    'tl.constexpr', H: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT:
    'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if HEAD_FIRST:
        b_g = tl.load(g + i_bh * T + i_t * BT + tl.arange(0, BT))
        b_g_last = tl.load(g + i_bh * T + min(i_t * BT + BT, T) - 1)
    else:
        b_g = tl.load(g + i_b * T * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            )
        b_g_last = tl.load(g + i_b * T * H + (min(i_t * BT + BT, T) - 1) *
            H + i_h)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * NT * K * V, (NT * K, V), (V, 1
            ), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh) * tl.exp(-b_g + b_g_last)[:, None]
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T * K, (K, T), (1, K), (i_k *
                BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + i_b * T * H * K + i_h * K, (K, T),
                (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    b_A = b_A * tl.exp(b_g[None, :] - b_g[:, None]) * scale
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :],
        b_A, 0)
    if HEAD_FIRST:
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * T * V, (T, V), (V, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
    else:
        p_do = tl.make_block_ptr(do + i_b * T * H * V + i_h * V, (T, V), (H *
            V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_b * T * H * V + i_h * V, (T, V), (H *
            V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A, b_do)
    tl.store(p_dv, b_dv, boundary_check=(0, 1))

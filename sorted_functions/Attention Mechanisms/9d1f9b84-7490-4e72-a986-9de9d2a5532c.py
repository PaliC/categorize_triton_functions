import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_bwd_dv_kernel(q, k, g, do, dv, dh, s_k_h, s_k_t, s_v_h, s_v_t,
    s_h_h, s_h_t, T, K, V, scale, BT: 'tl.constexpr', BK: 'tl.constexpr',
    BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    last_idx = min(i_t * BT + BT, T) - 1
    b_g = tl.load(g + i_bh * T + i_t * BT + tl.arange(0, BT))
    b_g_last = tl.load(g + i_bh * T + last_idx)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1),
            (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh) * tl.exp(-b_g + b_g_last)[:, None]
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k *
            BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    b_A = b_A * tl.exp(b_g[None, :] - b_g[:, None]) * scale
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :],
        b_A, 0)
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t *
        BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t *
        BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A, b_do)
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t *
        BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))

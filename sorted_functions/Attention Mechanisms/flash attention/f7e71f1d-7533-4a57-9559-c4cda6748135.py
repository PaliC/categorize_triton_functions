import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_dv(k, g, A, do, dh, dv, T: 'tl.constexpr', H:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT:
    'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if HEAD_FIRST:
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, 
            i_t * BT), (BT, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * T * V, (T, V), (V, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
    else:
        p_A = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (BT, T), (
            1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_b * T * H * V + i_h * V, (T, V), (H *
            V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_b * T * H * V + i_h * V, (T, V), (H *
            V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :],
        b_A, 0.0)
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = tl.dot(b_A, b_do, allow_tf32=False)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
            p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * T * K + min(
                i_t * BT + BT, T) * K - K + o_k, BK), BK)
        else:
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_gn = tl.max_contiguous(tl.multiple_of(g + i_b * T * H * K + (
                min(i_t * BT + BT, T) - 1) * H * K + i_h * K + o_k, BK), BK)
        p_dh = tl.make_block_ptr(dh + i_bh * NT * K * V + i_t * K * V, (K,
            V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_gn = tl.exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = b_k * b_gn
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh)
    tl.store(p_dv, b_dv, boundary_check=(0, 1))

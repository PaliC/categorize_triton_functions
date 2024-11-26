import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_inter(q, k, v, h, g, do, dh, dq, dk, dq2, dk2, dg,
    scale, T: 'tl.constexpr', H: 'tl.constexpr', K: 'tl.constexpr', V:
    'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NT: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    if HEAD_FIRST:
        p_gk = tl.make_block_ptr(g + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * T * K + min(T, 
            i_t * BT + BT) * K - K + o_k, BK), BK)
    else:
        p_gk = tl.make_block_ptr(g + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_b * T * H * K + (min(
            T, i_t * BT + BT) - 1) * H * K + i_h * K + o_k, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)
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
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V + i_t * V * K, (V, K),
            (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * NT * K * V + i_t * V * K, (V,
            K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, b_dh)
    b_dgk *= tl.exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gn = tl.exp(b_gn[None, :] - b_gk)
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * b_gn
    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT,
            i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT,
            i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :
        ] + b_dgk[None, :]
    if HEAD_FIRST:
        p_dq = tl.make_block_ptr(dq2 + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk2 + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_dq = tl.make_block_ptr(dq2 + i_b * T * H * K + i_h * K, (T, K), (
            H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk2 + i_b * T * H * K + i_h * K, (T, K), (
            H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))

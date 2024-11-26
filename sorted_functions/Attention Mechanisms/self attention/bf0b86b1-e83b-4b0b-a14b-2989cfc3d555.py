import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_bwd_kernel_V(k, v, h, g, A, do, dh, dq, dk, dv, dA, scale, T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NG:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_t = min(i_t * BT + BT, T)
    o_k = i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bg * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bg * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T * K + (o_t - 1) *
        K + o_k, BK), BK)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t *
        BT), (BT, BT), (0, 1))
    m_k = o_k < K
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gn = tl.exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
    b_k = b_k * b_gn
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bg * T * V, (T, V), (V, 1), (i_t * BT,
            i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * NT * K * V + i_t * V * K, (V, K),
            (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * NT * K * V + i_t * K * V, (K,
            V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * T * V, (T, V),
            (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh = b_dh
        b_dv = tl.dot(b_k, b_dh)
        if i_k == 0:
            b_dv += tl.dot(b_A, b_do)
        b_do = b_do * scale
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, tl.trans(b_dh))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * b_gn
    p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
        BT, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_dA = tl.where(m_s, b_dA, 0.0)
    if i_k == 0:
        tl.store(p_dA, b_dA, boundary_check=(0, 1))

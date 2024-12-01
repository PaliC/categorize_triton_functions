import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_bwd_kernel_K(q, k, v, h, g, A, do, dh, dq, dk, dv, dA, scale,
    T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NG:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)
    m_s = o_i[:, None] >= o_i[None, :]
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bg * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT), (BT,
        1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_A = tl.dot(b_q * scale, tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.0)
    tl.store(p_A, b_A, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_v = tl.make_block_ptr(v + i_bg * T * V, (T, V), (V, 1), (i_t * BT,
            i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * NT * K * V + i_t * K * V, (V, K),
            (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_g = tl.make_block_ptr(g + i_bg * T * V, (T, V), (V, 1), (i_t * BT,
            i_v * BV), (BT, BV), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T * V + (o_t - 1
            ) * V + o_v, BV), BV)
        m_v = o_v < V
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * NT * K * V + i_t * K * V, (K,
            V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * T * V, (T, V),
            (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_v = b_v * tl.exp(b_gn[None, :] - b_g)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_h = b_h
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_g) * scale
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh = b_dh
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, tl.trans(b_dh))
        b_dv = tl.exp(b_gn[None, :] - b_g) * tl.dot(b_k, b_dh)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
        BT, 0), (BT, BT), (1, 0))
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    b_dq += tl.dot(b_dA, b_k)
    b_dk += tl.dot(tl.trans(b_dA), b_q)
    p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))

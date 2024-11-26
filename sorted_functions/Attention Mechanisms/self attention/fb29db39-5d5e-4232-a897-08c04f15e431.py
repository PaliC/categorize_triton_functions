import triton
import triton.language as tl
import torch

@triton.jit
def chunk_rwkv6_fwd_kernel_intra(q, k, g, gs, u, A, s_k_h, s_k_t, s_k_d,
    scale, H, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr', DK: 'tl.constexpr'
    ):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC
        ) % NC
    i_h = i_bh % H
    n_bh = tl.num_programs(2)
    o_k = i_k * BK + tl.arange(0, BK)
    o_q = i_t * BT + i_i * BC
    m_k = o_k < K
    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
            (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
            (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT),
            (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        b_gn = tl.load(g + i_bh * T * K + (o_q - 1) * K + o_k, mask=m_k & (
            i_i > 0) & (o_q <= T), other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_gs - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A = tl.dot(b_qg, b_kg, allow_tf32=False)
        tl.store(p_A, b_A, boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
            (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t *
            BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_q_u = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        o_i = tl.arange(0, BC)
        o_g = i_bh * T * K + (i_t * BT + i_j * BC) * K + o_k
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.
            arange(0, BC)) * BT + i_j * BC
        m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
        p_u = tl.make_block_ptr(u + i_h * DK, (DK,), (1,), i_k * BK, (BK,),
            (0,))
        b_u = tl.load(p_u, boundary_check=(0,))
        for j in range(0, BC):
            b_k = tl.load(p_k, boundary_check=(0,))
            b_gk = tl.load(g + o_g + j * K, mask=m_k & (i_t * BT + i_j * BC +
                j < T), other=0)
            b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_gs - b_gk[None, :]) *
                scale, 1)
            b_A = tl.where(o_i > j, b_A, 0.0)
            b_q_u = tl.load(p_q_u, boundary_check=(0,))
            b_A_u = tl.sum(b_q_u * b_k * b_u * scale, axis=0)
            m_u = tl.arange(0, BC) == j
            b_A = tl.where(m_u, b_A_u, b_A)
            tl.store(A + o_A + j, b_A, mask=m_A)
            p_k = tl.advance(p_k, (K,))
            p_q_u = tl.advance(p_q_u, (K,))

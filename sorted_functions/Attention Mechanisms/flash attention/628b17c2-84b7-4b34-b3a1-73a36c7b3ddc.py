import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_intra_V(q, k, g, A, s_k_h, s_k_t, s_k_d, scale, T:
    'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC
        ) % NC
    n_bh = tl.num_programs(2)
    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
            (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
        p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT),
            (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        b_gn = tl.load(p_gn, boundary_check=(0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A = tl.dot(b_qg, b_kg, allow_tf32=False)
        tl.store(p_A, b_A, boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t *
            BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.
            arange(0, BC)) * BT + i_j * BC
        m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
        for j in range(0, BC):
            b_k = tl.load(p_k, boundary_check=(0,))
            b_gk = tl.load(p_gk, boundary_check=(0,))
            b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]) *
                scale, 1)
            b_A = tl.where(o_i >= j, b_A, 0.0)
            tl.store(A + o_A + j, b_A, mask=m_A)
            p_k = tl.advance(p_k, (K,))
            p_gk = tl.advance(p_gk, (K,))

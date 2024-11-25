import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'NC', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_intra(q, k, g, dA, dq, dk, s_k_h, s_k_t, T:
    'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT +
        i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t *
            BT + i_i * BC) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (
                i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1),
                (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1),
                (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * tl.exp(b_gn[None, :] - b_gk)
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * BT + i_i * BC
    m_dA = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    p_kj = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + 
        i_i * BC) * K + o_k, BK), BK)
    p_gkj = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT +
        i_i * BC) * K + o_k, BK), BK)
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, mask=m_k, other=0)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0)
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_g -
            b_gkj[None, :]), 0.0)
        p_kj += K
        p_gkj += K
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t *
        BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT +
        i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t *
        BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t *
            BT + i_i * BC + BC - 1) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (
                i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (
                i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (BT, T), (1, BT),
                (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_qg = b_q * tl.exp(b_g - b_gn[None, :])
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= tl.exp(b_gn[None, :] - b_gk)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(
        0, BC)
    p_qj = tl.max_contiguous(tl.multiple_of(q + i_bh * s_k_h + (i_t * BT + 
        i_i * BC) * K + o_k, BK), BK)
    p_gqj = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT +
        i_i * BC) * K + o_k, BK), BK)
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j * BT)
        b_qj = tl.load(p_qj, mask=m_k, other=0)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0)
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[
            None, :] - b_gk), 0.0)
        p_qj += K
        p_gqj += K
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t *
        BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))

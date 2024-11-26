import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_fwd_kernel_intra_Vk(q, k, g, A, i_k, i_c, i_bh, scale, T:
    'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr'
    ):
    i_bg = i_bh // NG
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC
        ) % NC
    o_k = i_k * BK + tl.arange(0, BK)
    if i_t * BT + i_i * BC > T:
        return
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT +
        i_i * BC, i_j * BC), (BC, BC), (1, 0))
    b_A = tl.zeros([BC, BC], tl.float32)
    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT +
            i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * T * K, (T, K), (K, 1), (i_t * BT +
            i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bg * T * K, (K, T), (1, K), (i_k * BK,
            i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bg * T * K, (K, T), (1, K), (i_k *
            BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T * K + min(i_t *
            BT + i_i * BC, T) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=o_k < K, other=0.0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A = tl.dot(b_qg, b_kg)
        if i_k != 0:
            b_A += tl.load(p_A, boundary_check=(0, 1))
        tl.store(p_A, b_A, boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT +
            i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * T * K, (T, K), (K, 1), (i_t * BT +
            i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bg * T * K + (i_t * BT +
            i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bg * T * K + (i_t *
            BT + i_j * BC) * K + o_k, BK), BK)
        m_k = o_k < K
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        o_i = tl.arange(0, BC)
        m_A = o_i[:, None] >= o_i[None, :]
        for j in range(0, min(BC, T - i_t * BT - i_j * BC)):
            b_k = tl.load(p_k, mask=m_k, other=0.0)
            b_gk = tl.load(p_gk, mask=m_k, other=0.0)
            b_Aj = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]) *
                scale, 1)
            b_A = tl.where((o_i == j)[None, :], b_Aj[:, None], b_A)
            p_k += K
            p_gk += K
        b_A = tl.where(m_A, b_A, 0.0)
        if i_k != 0:
            b_A += tl.load(p_A, boundary_check=(0, 1))
        tl.store(p_A, b_A, boundary_check=(0, 1))
    elif i_k == 0:
        tl.store(p_A, b_A, boundary_check=(0, 1))

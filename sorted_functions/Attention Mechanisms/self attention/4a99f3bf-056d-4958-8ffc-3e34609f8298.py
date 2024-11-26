import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BT'])
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(q, k, g, A, scale, T:
    'tl.constexpr', H: 'tl.constexpr', K: 'tl.constexpr', BT:
    'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', HEAD_FIRST:
    'tl.constexpr'):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    if HEAD_FIRST:
        o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
            ) * BT + i_j * BC
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT +
            i_i * BC, 0), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * T * K, (T, K), (K, 1), (i_t * BT +
            i_i * BC, 0), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * T * K + (i_t * BT +
            i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * T * K + (i_t *
            BT + i_j * BC) * K + o_k, BK), BK)
    else:
        o_A = i_b * T * H * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
            ) * H * BT + i_h * BT + i_j * BC
        p_q = tl.make_block_ptr(q + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_b * T * H * K + (i_t *
            BT + i_j * BC) * H * K + i_h * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_b * T * H * K + (i_t *
            BT + i_j * BC) * H * K + i_h * K + o_k, BK), BK)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0)
        b_gk = tl.load(p_gk, mask=m_k, other=0)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K if HEAD_FIRST else H * K
        p_gk += K if HEAD_FIRST else H * K

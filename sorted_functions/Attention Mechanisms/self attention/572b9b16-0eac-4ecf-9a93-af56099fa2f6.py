import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BC', 'BK'])
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(q, k, g, A, scale, T:
    'tl.constexpr', H: 'tl.constexpr', K: 'tl.constexpr', BT:
    'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC:
    'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (i_k *
                BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gk = tl.make_block_ptr(g + i_bh * T * K, (K, T), (1, K), (i_k *
                BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * T * K + (i_t *
                BT + i_i * BC) * K + o_k, BK), BK)
        else:
            p_q = tl.make_block_ptr(q + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (K, T),
                (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gk = tl.make_block_ptr(g + i_b * T * H * K + i_h * K, (K, T),
                (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(g + i_b * T * H * K + (
                i_t * BT + i_i * BC) * H * K + i_h * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)
    if HEAD_FIRST:
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    else:
        p_A = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (T, BT), (
            H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A, boundary_check=(0, 1))

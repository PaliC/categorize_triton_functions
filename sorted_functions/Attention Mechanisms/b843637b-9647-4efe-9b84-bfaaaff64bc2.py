import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BT'])
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra(q, k, gi, ge, u, A, s_k_h,
    s_k_t, s_k_d, scale, H: 'tl.constexpr', T: 'tl.constexpr', K:
    'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK:
    'tl.constexpr', NC: 'tl.constexpr'):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_i * BC >= T:
        return
    i_j = i_i
    i_h = i_bh % H
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * BT + i_j * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    i_k = 0
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_u = tl.make_block_ptr(u + i_h * s_k_t, (s_k_t,), (1,), i_k * BK, (BK,
        ), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t *
            BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_k = tl.load(p_k, boundary_check=(0,))
        b_gk = tl.load(p_gk, boundary_check=(0,))
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.0)
        p_qj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qj = tl.load(p_qj, boundary_check=(0,))
        p_qi = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qi = tl.load(p_qi, boundary_check=(0,))
        A_jj = tl.sum(b_qi * b_k * b_u * scale)
        b_A = tl.where(o_i != j, b_A, A_jj)
        tl.store(A + o_A + j, b_A, mask=m_A)

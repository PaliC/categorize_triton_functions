import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BC', 'BK'])
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_inter(q, k, gi, ge, A, s_k_h, s_k_t,
    s_k_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr',
    BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_i <= i_j:
        return
    if i_t * BT + i_i * BC >= T:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gq = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
            (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
            (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), ((
            i_t * BT + i_j * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
        b_gn = tl.load(p_gn, boundary_check=(0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.load(p_gq, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_gq - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT +
        i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A, boundary_check=(0, 1))

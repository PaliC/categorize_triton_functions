import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_rwkv6_bwd_kernel_inter(q, k, v, h, gi, ge, u, do, dh, dA, dq, dk,
    dq2, dk2, dg, du, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h,
    s_h_t, s_h_d, scale, H: 'tl.constexpr', T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    n_bh = tl.num_programs(2)
    last_idx = min(T, i_t * BT + BT) - 1
    p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), (
        last_idx * K + i_k * BK,), (BK,), (0,))
    b_gn = tl.load(p_gn, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (
            s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * V * K, (V, K), (
            s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, b_dh)
    p_gk = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dgk *= tl.exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    p_gi = tl.make_block_ptr(gi + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * tl.exp(b_gn[None, :] - b_gi)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :
        ] + b_dgk[None, :] - b_q * b_dq
    o_i = tl.arange(0, BT)
    p_dA_dig = dA + i_bh * T * BT + (i_t * BT + o_i) * BT + o_i
    b_dA_dig = tl.load(p_dA_dig, mask=i_t * BT + o_i < T, other=0)
    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    b_dq += b_dA_dig[:, None] * b_u[None, :] * b_k
    b_dk += b_dA_dig[:, None] * b_u[None, :] * b_q
    b_du = tl.sum(b_dA_dig[:, None] * b_q * b_k, axis=0)
    p_du = tl.make_block_ptr(du + (i_h + i_t * n_bh) * K, (K,), (1,), (i_k *
        BK,), (BK,), (0,))
    tl.store(p_du, b_du, boundary_check=(0,))
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq2 + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2 + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))

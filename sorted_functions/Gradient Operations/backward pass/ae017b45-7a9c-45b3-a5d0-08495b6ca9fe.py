import triton
import triton.language as tl
import torch

@triton.jit
def post_process_grad(q, k, v, u, do, dk, dq, du, scale, s_k_h, s_k_t,
    s_k_d, s_v_h, s_v_t, s_v_d, H, T: 'tl.constexpr', BT: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, 0), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, 0), (BT, BK), (1, 0))
    p_du = tl.make_block_ptr(du + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT, 0), (BT, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
        i_t * BT, 0), (BT, BV), (1, 0))
    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (0,), (BK,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0,))
    b_vdo = tl.sum(b_v * b_do, axis=1)
    b_du = b_vdo[:, None] * b_k * b_q * scale
    b_dq = b_vdo[:, None] * b_k * b_u[None, :] * scale
    b_dk = b_vdo[:, None] * b_q * b_u[None, :] * scale
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_du, b_du, boundary_check=(0, 1))

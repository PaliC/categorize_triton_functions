import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_K(q, k, v, z, h, A, do, dh, dq, dk, dv, dA, s_k_h,
    s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT), (BT,
        1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_A = tl.dot(b_q * scale, tl.trans(b_k), allow_tf32=False)
    b_A = tl.where(m_s, b_A, 0.0)
    tl.store(p_A, b_A, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_zp = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), (i_p *
            V + i_v * BV,), (BV,), (0,))
        p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((
            i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (V, K), (
            s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_v_h, (T, V),
            (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_zp = tl.load(p_zp, boundary_check=(0,))
        b_zc = tl.load(p_zc, boundary_check=(0,))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.exp(b_v - b_zc[None, :])
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_z = tl.exp(b_zp[None, :] - b_z)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * b_z * scale
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = b_v * tl.dot(b_k, b_dh, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
        BT, 0), (BT, BT), (1, 0))
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    b_dq += tl.dot(b_dA, b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_dA), b_q, allow_tf32=False)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))

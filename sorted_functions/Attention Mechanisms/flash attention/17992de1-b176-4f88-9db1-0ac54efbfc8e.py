import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_K(q, k, h, g, o, A, s_k_h, s_k_t, s_k_d, s_v_h,
    s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_A += tl.dot(b_q, b_k, allow_tf32=False)
    p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT, i_v * BV), (BT, BV), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o = b_o * tl.exp(b_g)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT,
        0), (BT, BT), (1, 0))
    b_A = tl.where(m_s, b_A, 0.0)
    if i_v == 0:
        tl.store(p_A, b_A, boundary_check=(0, 1))

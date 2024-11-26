import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'K', 'V'])
@triton.jit
def chunk_transform_qk_fwd_kernel(q, k, v, beta, o, A, q_new, k_new,
    A_local, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, scale, T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', BT: 'tl.constexpr',
    OUTPUT_ATTENTIONS: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t *
        BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT, 0), (BT, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)) * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    p_T = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT,
        0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k), allow_tf32=False), 0)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (
        BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_k_beta = b_k * b_beta[:, None]
    b_qkT = tl.dot(b_qk, b_T, allow_tf32=False)
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1),
            (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_a, b_qkT, boundary_check=(0, 1))
    b_kkT = tl.dot(b_kk, b_T, allow_tf32=False)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, tl.dot(b_qkT, b_v), boundary_check=(0, 1))
    p_q_new = tl.make_block_ptr(q_new + i_bh * s_k_h, (T, K), (s_k_t, s_k_d
        ), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, b_q - tl.dot(b_qkT, b_k_beta, allow_tf32=False),
        boundary_check=(0, 1))
    p_k_new = tl.make_block_ptr(k_new + i_bh * s_k_h, (T, K), (s_k_t, s_k_d
        ), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_k_new, b_k - tl.dot(tl.trans(b_kkT), b_k_beta, allow_tf32=
        False), boundary_check=(0, 1))

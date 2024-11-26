import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_fwd_kernel_V(q, v, g, h, o, A, scale, T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NG: 'tl.constexpr', NT: 'tl.constexpr'
    ):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT,
            i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * T * K, (T, K), (K, 1), (i_t * BT,
            i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * NT * K * V + i_t * K * V, (K, V),
            (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h)
    p_v = tl.make_block_ptr(v + i_bg * T * V, (T, V), (V, 1), (i_t * BT, 
        i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 
        i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT,
        0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o, boundary_check=(0, 1))

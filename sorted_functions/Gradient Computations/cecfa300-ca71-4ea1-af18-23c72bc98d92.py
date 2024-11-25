import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_rcum(s, r, c, o, s_sk_h, s_sk_t, s_sk_m, T, BT:
    'tl.constexpr', BM: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'
    ):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (
        (NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (
        (NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (
        (NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_t = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BM], dtype=tl.float32)
    for i in range(NT):
        p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m
            ,), ((NT - i) % NT * DM + i_m * BM,), (BM,), (0,))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_r = tl.load(p_r, boundary_check=(0,))
        b_c = tl.load(p_c, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))
        b_z = b_z * b_r
        b_o -= b_c * (b_z[None, :] + tl.dot(m_t, b_s, allow_tf32=False))
        b_z += tl.sum(b_s, 0)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_s = tl.advance(p_s, (-BT, 0))
        p_c = tl.advance(p_c, (-BT, 0))
        p_o = tl.advance(p_o, (-BT, 0))

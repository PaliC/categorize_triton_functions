import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_s(q, k, s, rk, ck, pk, s_qk_h, s_qk_t, s_qk_d,
    s_sk_h, s_sk_t, s_sk_m, T, scale, BT: 'tl.constexpr', BK:
    'tl.constexpr', BM: 'tl.constexpr', DK: 'tl.constexpr', DM:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_m, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (
        i_k * BK, 0), (BK, BT), (0, 1))
    p_s = tl.make_block_ptr(s + (i_k * n_bh + i_bh) * s_sk_h, (T, DM), (
        s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,),
        (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m),
        (0, i_m * BM), (BT, BM), (1, 0))
    p_pk = tl.make_block_ptr(pk + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m),
        (0, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_hk = tl.zeros([BK, BM], dtype=tl.float32)
    for _ in range(NT):
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_rk = tl.load(p_rk, boundary_check=(0,))
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1))
        b_inter = tl.dot(b_q, b_hk, allow_tf32=False) * b_rk[None, :]
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_q, b_k, allow_tf32=False), 
            0), b_ck, allow_tf32=False)
        b_s = (b_inter + b_intra) * b_pk
        b_hk = b_hk * b_rk[None, :] + tl.dot(b_k, b_ck, allow_tf32=False)
        tl.store(p_s, b_s, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_s = tl.advance(p_s, (BT, 0))
        p_rk = tl.advance(p_rk, (DM,))
        p_ck = tl.advance(p_ck, (BT, 0))
        p_pk = tl.advance(p_pk, (BT, 0))

import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_dk(q, k, rk, ck, ds, dk, dsk, s_qk_h, s_qk_t,
    s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BK:
    'tl.constexpr', BM: 'tl.constexpr', DK: 'tl.constexpr', DM:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        (NT - 1) * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (
        i_k * BK, (NT - 1) * BT), (BK, BT), (0, 1))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,),
        (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m),
        ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t),
        (i_m * BM, (NT - 1) * BT), (BM, BT), (0, 1))
    p_dk = tl.make_block_ptr(dk + (i_m * n_bh + i_bh) * s_qk_h, (T, DK), (
        s_qk_t, s_qk_d), ((NT - 1) * BT, i_k * BK), (BT, BK), (1, 0))
    p_dsk = tl.make_block_ptr(dsk + (i_k * n_bh + i_bh) * s_sk_h, (T, DM),
        (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s, m_t = o_i[:, None] <= o_i[None, :], o_i[:, None] >= o_i[None, :]
    b_dhk = tl.zeros([BM, BK], dtype=tl.float32)
    for i in range(NT):
        p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (
            s_sk_m,), ((NT - i) % NT * DM + i_m * BM,), (BM,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_rk = tl.load(p_rk, boundary_check=(0,))
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_inter = tl.dot(b_ck * b_rk[None, :], b_dhk, allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_ck, b_ds, allow_tf32=False),
            0.0), b_q, allow_tf32=False)
        b_dk = b_inter + b_intra
        b_inter = tl.dot(b_dhk, b_k, allow_tf32=False) * b_rk[:, None]
        b_intra = tl.dot(b_ds, tl.where(m_t, tl.dot(b_q, b_k, allow_tf32=
            False), 0.0), allow_tf32=False)
        b_dsk = b_ck * tl.trans(b_inter + b_intra)
        b_dhk = b_dhk * b_rk[:, None] + tl.dot(b_ds, b_q, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dsk, b_dsk, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (-BT, 0))
        p_k = tl.advance(p_k, (0, -BT))
        p_ck = tl.advance(p_ck, (-BT, 0))
        p_ds = tl.advance(p_ds, (0, -BT))
        p_dk = tl.advance(p_dk, (-BT, 0))
        p_dsk = tl.advance(p_dsk, (-BT, 0))

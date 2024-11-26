import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_dq(k, rk, ck, dq, ds, s_qk_h, s_qk_t, s_qk_d,
    s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BK: 'tl.constexpr', BM:
    'tl.constexpr', DK: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'
    ):
    i_k, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        0, i_k * BK), (BT, BK), (1, 0))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,),
        (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t),
        (i_m * BM, 0), (BM, BT), (0, 1))
    p_dq = tl.make_block_ptr(dq + (i_m * n_bh + i_bh) * s_qk_h, (T, DK), (
        s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m),
        (0, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_hk = tl.zeros([BM, BK], dtype=tl.float32)
    for _ in range(NT):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_rk = tl.load(p_rk, boundary_check=(0,))
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_inter = tl.dot(b_ds * b_rk[None, :], b_hk, allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_ds, b_ck, allow_tf32=False),
            0), b_k, allow_tf32=False)
        b_dq = b_inter + b_intra
        b_hk = b_hk * b_rk[:, None] + tl.dot(b_ck, b_k, allow_tf32=False)
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
        p_k = tl.advance(p_k, (BT, 0))
        p_rk = tl.advance(p_rk, (DM,))
        p_ck = tl.advance(p_ck, (0, BT))
        p_dq = tl.advance(p_dq, (BT, 0))
        p_ds = tl.advance(p_ds, (BT, 0))

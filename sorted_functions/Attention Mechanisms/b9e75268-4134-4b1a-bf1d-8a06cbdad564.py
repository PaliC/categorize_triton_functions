import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_dp(v, rv, cv, pv, do, dp, s_qk_h, s_qk_t, s_qk_d,
    s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BV: 'tl.constexpr', BM:
    'tl.constexpr', DV: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'
    ):
    i_m, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (DV, T), (s_qk_d, s_qk_t), (
        i_v * BV, 0), (BV, BT), (0, 1))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,),
        (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m),
        (0, i_m * BM), (BT, BM), (1, 0))
    p_pv = tl.make_block_ptr(pv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m),
        (0, i_m * BM), (BT, BM), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d),
        (0, i_v * BV), (BT, BV), (1, 0))
    p_dp = tl.make_block_ptr(dp + (i_v * n_bh + i_bh) * s_sk_h, (T, DM), (
        s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_hv = tl.zeros([BV, BM], dtype=tl.float32)
    for _ in range(NT):
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_rv = tl.load(p_rv, boundary_check=(0,))
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        b_pv = tl.load(p_pv, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_inter = tl.dot(b_do, b_hv, allow_tf32=False) * b_rv[None, :]
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_do, b_v, allow_tf32=False),
            0), b_cv, allow_tf32=False)
        b_dp = (b_inter + b_intra) * b_pv
        b_hv = b_hv * b_rv[None, :] + tl.dot(b_v, b_cv, allow_tf32=False)
        tl.store(p_dp, b_dp, boundary_check=(0, 1))
        p_v = tl.advance(p_v, (0, BT))
        p_rv = tl.advance(p_rv, (DM,))
        p_cv = tl.advance(p_cv, (BT, 0))
        p_pv = tl.advance(p_pv, (BT, 0))
        p_do = tl.advance(p_do, (BT, 0))
        p_dp = tl.advance(p_dp, (BT, 0))

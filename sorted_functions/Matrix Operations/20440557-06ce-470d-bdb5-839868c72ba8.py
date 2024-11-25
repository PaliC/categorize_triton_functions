import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_dv(do, v, rv, cv, p, dv, dsv, s_qk_h, s_qk_t,
    s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BV:
    'tl.constexpr', BM: 'tl.constexpr', DV: 'tl.constexpr', DM:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_v, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_do = tl.make_block_ptr(do + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d),
        ((NT - 1) * BT, i_v * BV), (BT, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (DV, T), (s_qk_d, s_qk_t), (
        i_v * BV, (NT - 1) * BT), (BV, BT), (0, 1))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,),
        (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m),
        ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (
        i_m * BM, (NT - 1) * BT), (BM, BT), (0, 1))
    p_dv = tl.make_block_ptr(dv + (i_m * n_bh + i_bh) * s_qk_h, (T, DV), (
        s_qk_t, s_qk_d), ((NT - 1) * BT, i_v * BV), (BT, BV), (1, 0))
    p_dsv = tl.make_block_ptr(dsv + (i_v * n_bh + i_bh) * s_sk_h, (T, DM),
        (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s, m_t = o_i[:, None] <= o_i[None, :], o_i[:, None] >= o_i[None, :]
    b_dhv = tl.zeros([BM, BV], dtype=tl.float32)
    for i in range(NT):
        p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (
            s_sk_m,), ((NT - i) % NT * DM + i_m * BM,), (BM,), (0,))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_rv = tl.load(p_rv, boundary_check=(0,))
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_inter = tl.dot(b_cv * b_rv[None, :], b_dhv, allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_cv, b_p, allow_tf32=False),
            0.0), b_do, allow_tf32=False)
        b_dv = b_inter + b_intra
        b_inter = tl.dot(b_dhv, b_v, allow_tf32=False) * b_rv[:, None]
        b_intra = tl.dot(b_p, tl.where(m_t, tl.dot(b_do, b_v, allow_tf32=
            False), 0.0), allow_tf32=False)
        b_dsv = b_cv * tl.trans(b_inter + b_intra)
        b_dhv = b_dhv * b_rv[:, None] + tl.dot(b_p, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
        tl.store(p_dsv, b_dsv, boundary_check=(0, 1))
        p_do = tl.advance(p_do, (-BT, 0))
        p_v = tl.advance(p_v, (0, -BT))
        p_cv = tl.advance(p_cv, (-BT, 0))
        p_p = tl.advance(p_p, (0, -BT))
        p_dv = tl.advance(p_dv, (-BT, 0))
        p_dsv = tl.advance(p_dsv, (-BT, 0))

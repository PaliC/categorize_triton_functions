import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_retention_bwd_kernel_dqkv(q, k, v, h, do, dh, dq, dk, dv, s_qk_h,
    s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H:
    'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT:
    'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    n_bh = tl.num_programs(2)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b
        )
    d_q = d_q * scale
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0
        ) * scale
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (
        i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t),
            (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t,
            s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1),
            (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_vo_h, (T, V),
            (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s,
            b_do, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    b_ds = b_ds * d_s
    b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))

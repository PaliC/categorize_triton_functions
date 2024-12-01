import triton
import triton.language as tl
import torch

@triton.jit
def fused_chunk_retention_bwd_kernel(q, k, v, do, dq, dk, dv, h0, s_k_h,
    s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, scale, B: 'tl.constexpr', H:
    'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b) * scale, tl.math.exp2((BT -
        o_i - 1) * b_b)
    d_b = tl.math.exp2(BT * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0
        ) * scale
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (V, K), (1, V), (i_v *
            BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (
            i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_k_h, (T, K),
            (s_k_t, s_k_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = b_do * d_q[:, None]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = b_ds * d_s
        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)
        if CHECK and i == 0:
            b_dq += tl.dot(b_dd, b_h, allow_tf32=False)
            b_h = d_b * b_h + tl.dot(b_v * d_k[None, :], b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_dd, b_h, allow_tf32=False)
            b_h = d_b * b_h + tl.dot(b_v * d_k[None, :], b_k, allow_tf32=False)
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    d_s = tl.trans(d_s)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (
            T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_k_h, (T, K),
            (s_k_t, s_k_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_v_h, (T, V),
            (s_v_t, s_v_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = b_do * d_q[:, None]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = b_ds * d_s
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * d_k[:, None
                ]
            b_dv += tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * d_k[:, None
                ]
            b_dv += tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))

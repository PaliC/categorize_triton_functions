import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def fused_chunk_delta_rule_bwd_kernel(q, k, v, d, dht, dh0, do, dq, dk, dv,
    dd, initial_state, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, B, H, T,
    scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK:
    'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    USE_DHT: 'tl.constexpr', USE_DHO: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_dht = tl.make_block_ptr(dht + i_bh * DK * DV, (DK, DV), (DV, 1),
            (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))
    m_s = o_i[:, None] <= o_i[None, :]
    for i in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (DK, T), (s_k_d, s_k_t),
            (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_d = tl.make_block_ptr(d + i_bh * s_k_h, (DK, T), (s_k_d, s_k_t),
            (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, DK), (s_k_t, s_k_d),
            (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, DV), (s_v_t, s_v_d),
            (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, DV), (s_v_t, s_v_d),
            (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_k_h, (T, DK),
            (s_k_t, s_k_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_v_h, (T, DV),
            (s_v_t, s_v_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv += tl.dot(b_k, b_dh, allow_tf32=False)
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        b_dh -= tl.dot(b_d, b_dv, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    if USE_DHO:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * DK * DV, (DK, DV), (DV, 1),
            (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh, boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (
            1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    NT = tl.cdiv(T, BT)
    for i in range(0, NT):
        p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, DV), (s_v_t, s_v_d),
            (i * BT, i_v * BV), (BT, BV), (1, 0))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dd = tl.dot(b_dv, b_h, allow_tf32=False)
        p_dd = tl.make_block_ptr(dd + (i_bh + i_v * B * H) * s_k_h, (T, DK),
            (s_k_t, s_k_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dd, -b_dd, boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, DK), (s_k_t, s_k_d),
            (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (DV, T), (s_v_d, s_v_t),
            (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, DV), (s_v_t, s_v_d),
            (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_k_h, (T, DK),
            (s_k_t, s_k_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq, boundary_check=(0, 1))

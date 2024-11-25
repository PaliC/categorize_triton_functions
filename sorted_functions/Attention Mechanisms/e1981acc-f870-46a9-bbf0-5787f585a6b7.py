import triton
import triton.language as tl
import torch

@triton.jit
def fused_chunk_based_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h,
    s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H:
    'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h_1o = tl.zeros([BV, BK], dtype=tl.float32)
    b_h_2o = tl.zeros([BV, BK * BK], dtype=tl.float32)
    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
            (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
            (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t),
            (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t,
            s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, K),
            (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dz = dz + i_bh * T + tl.arange(0, BT) + i * BT
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=tl.arange(0, BT) + i * BT < T)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dq += tl.dot(b_do, b_h_1o, allow_tf32=False)
        if i_v == 0:
            b_dq += b_dz[:, None] * k_1o
        b_dq_2o = tl.dot(b_do, b_h_2o, allow_tf32=False) * 0.5
        if i_v == 0:
            b_dq_2o += b_dz[:, None] * k_2o * 0.5
        b_dq_2o = tl.reshape(b_dq_2o, [BT, BK, BK])
        b_dq += tl.sum(b_dq_2o * b_q[:, :, None], axis=1)
        b_dq += tl.sum(b_dq_2o * b_q[:, None, :], axis=2)
        b_dq *= scale
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot(b_ds * (1 + b_s), b_k, allow_tf32=False)
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK])
        b_h_2o = b_h_2o + tl.dot(b_v, b_k_2o, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_v, b_k, allow_tf32=False)
        if i_v == 0:
            k_1o += tl.sum(b_k, axis=0)[None, :]
            k_2o += tl.sum(b_k_2o, axis=0)[None, :]
    tl.debug_barrier()
    b_h_1o = None
    b_h_2o = None
    b_dh_1o = tl.zeros([BK, BV], dtype=tl.float32)
    b_dh_2o = tl.zeros([BK * BK, BV], dtype=tl.float32)
    b_dh_0o = tl.zeros([BV], dtype=tl.float32)
    m_s = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]
    dq_1o = tl.zeros([1, BK], dtype=tl.float32)
    dq_2o = tl.zeros([BK * BK, 1], dtype=tl.float32)
    for i in range(tl.cdiv(T, BT) * BT - BT, -BT, -BT):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
            (i_k * BK, i), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
            (i, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
            (i, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t,
            s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, K),
            (s_qk_t, s_qk_d), (i, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, V),
            (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_dz = dz + i_bh * T + tl.arange(0, BT) + i
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=tl.arange(0, BT) + i < T)
        b_q = b_q * scale
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds *= 1 + b_s
        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s2, b_do, allow_tf32=False)
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK])
        b_dv += tl.dot(b_k, b_dh_1o, allow_tf32=False)
        b_dv += tl.dot(b_k_2o, b_dh_2o, allow_tf32=False)
        b_dv += b_dh_0o
        b_dk += tl.dot(b_v, tl.trans(b_dh_1o), allow_tf32=False)
        if i_v == 0:
            b_dk += dq_1o
        b_dk_2o = tl.dot(b_dh_2o, tl.trans(b_v), allow_tf32=False)
        if i_v == 0:
            b_dk_2o += dq_2o
        b_dk_2o = tl.reshape(b_dk_2o, [BK, BK, BT])
        b_k_fp32 = tl.trans(b_k)
        b_dk2 = tl.sum(b_dk_2o * b_k_fp32[:, None, :], axis=0)
        b_dk2 += tl.sum(b_dk_2o * b_k_fp32[None, :, :], axis=1)
        b_dk += tl.trans(b_dk2)
        b_dh_0o += tl.sum(b_do, axis=0)
        b_dh_1o = b_dh_1o + tl.dot(b_q, b_do, allow_tf32=False)
        b_q_2o = b_q[None, :, :] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BK * BK, BT])
        b_dh_2o = b_dh_2o + tl.dot(b_q_2o, b_do, allow_tf32=False) * 0.5
        if i_v == 0:
            dq_1o += tl.sum(b_dz[None, :] * b_q, axis=1)[None, :]
            dq_2o += (tl.sum(b_dz[None, :] * b_q_2o, axis=1) * 0.5)[:, None]
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))

import triton
import triton.language as tl
import torch

@triton.jit
def parallel_retention_bwd_kernel_dkv(i_bh, i_t, i_k, i_v, i_h, q, k, v, do,
    dk, dv, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr',
    K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BS)
    p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 
        i_v * BV), (BT, BV), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    NTS = tl.cdiv(T, BS)
    d_h = tl.math.exp2((BT - tl.arange(0, BT)) * b_b)
    b_kd = b_k * d_h[:, None]
    d_q = tl.math.exp2(tl.arange(0, BS) * b_b)
    for i in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i, i_k *
            BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i, i_v *
            BV), (BS, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * d_q[:, None]
        b_dk *= d_b
        b_dv *= d_b
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_s = tl.dot(b_kd, tl.trans(b_q), allow_tf32=False)
        b_dk += tl.dot(b_ds, b_q, allow_tf32=False)
        b_dv += tl.dot(b_s, b_do, allow_tf32=False)
    b_dk *= d_h[:, None] * scale
    b_dv *= scale
    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BS), tl.arange(0, BT)
    for i in range(i_t * BT, (i_t + 1) * BT, BS):
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i, i_k *
            BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i, i_v *
            BV), (BS, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.math.exp2((-o_k[:, None] + o_q[None, :]) *
            b_b), 0) * scale
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False) * d_s
        b_s = tl.dot(b_k, tl.trans(b_q), allow_tf32=False) * d_s
        b_dk += tl.dot(b_ds, b_q, allow_tf32=False)
        b_dv += tl.dot(b_s, b_do, allow_tf32=False)
        o_q += BS
    p_dk = tl.make_block_ptr(dk + (i_v * B * H + i_bh) * T * K, (T, K), (K,
        1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_k * B * H + i_bh) * T * V, (T, V), (V,
        1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))

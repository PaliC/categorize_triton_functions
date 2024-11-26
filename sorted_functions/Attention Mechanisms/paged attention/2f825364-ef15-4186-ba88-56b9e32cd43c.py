import triton
import triton.language as tl
import torch

@triton.jit
def parallel_retention_bwd_kernel_dq(i_bh, i_t, i_k, i_v, i_h, k, v, do, dq,
    scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (0, i_k * BK),
        (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (V, T), (1, V), (i_v * BV, 0),
        (BV, BS), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 
        i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BS)
    d_h = tl.math.exp2((BS - tl.arange(0, BS)) * b_b)
    for i in range(0, i_t * BT, BS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_h[None, :]
        if i != 0:
            b_dq *= d_b
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (0, BS))
    b_dq *= tl.math.exp2(tl.arange(0, BT) * b_b)[:, None] * scale
    o_q = tl.arange(0, BT)
    o_k = tl.arange(0, BS)
    p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 
        i_k * BK), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (V, T), (1, V), (i_v * BV, 
        i_t * BT), (BV, BS), (0, 1))
    for _ in range(i_t * BT, (i_t + 1) * BT, BS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) *
            b_b), 0)
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_s * scale
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (0, BS))
        o_k += BS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * T * K, (T, K), (K,
        1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))

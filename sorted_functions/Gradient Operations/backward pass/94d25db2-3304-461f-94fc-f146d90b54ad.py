import triton
import triton.language as tl
import torch

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None})
@triton.jit
def fused_recurrent_retention_bwd_kernel(q, k, v, h0, do, dq, dk, dv, dh0,
    dht, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    USE_FINAL_STATE_GRADIENT: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
    if HEAD_FIRST:
        p_q = q + i_bh * T * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_bh * T * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_bh * T * V + i_v * BV + tl.arange(0, BV)
        p_do = do + i_bh * T * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B * H + i_bh) * T * K + i_k * BK + tl.arange(0, BK)
    else:
        p_q = q + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B + i_b
            ) * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_do = tl.load(p_do, mask=mask_v, other=0)
        b_h = b_b * b_h + b_k[:, None] * b_v[None, :]
        b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq, mask=mask_k)
        p_k += K if HEAD_FIRST else H * K
        p_v += V if HEAD_FIRST else H * V
        p_do += V if HEAD_FIRST else H * V
        p_dq += K if HEAD_FIRST else H * K
    tl.debug_barrier()
    if HEAD_FIRST:
        p_q = q + i_bh * T * K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_k = k + i_bh * T * K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_v = v + i_bh * T * V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_do = do + i_bh * T * V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_dk = dk + (i_bh + i_v * B * H) * T * K + i_k * BK + tl.arange(0, BK
            ) + (T - 1) * K
        p_dv = dv + (i_bh + i_k * B * H) * T * V + i_v * BV + tl.arange(0, BV
            ) + (T - 1) * V
    else:
        p_q = q + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK) + (T
             - 1) * H * K
        p_k = k + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK) + (T
             - 1) * H * K
        p_v = v + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV) + (T
             - 1) * H * V
        p_do = do + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV) + (
            T - 1) * H * V
        p_dk = dk + (i_v * B + i_b
            ) * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK) + (T - 1
            ) * H * K
        p_dv = dv + (i_k * B + i_b
            ) * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1
            ) * H * V
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_ht, mask=mask_h, other=0)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_do = tl.load(p_do, mask=mask_v, other=0)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        b_dh *= b_b
        tl.store(p_dk, b_dk, mask=mask_k)
        tl.store(p_dv, b_dv, mask=mask_v)
        p_q -= K if HEAD_FIRST else H * K
        p_k -= K if HEAD_FIRST else H * K
        p_v -= V if HEAD_FIRST else H * V
        p_do -= V if HEAD_FIRST else H * V
        p_dk -= K if HEAD_FIRST else H * K
        p_dv -= V if HEAD_FIRST else H * V
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh, mask=mask_h)

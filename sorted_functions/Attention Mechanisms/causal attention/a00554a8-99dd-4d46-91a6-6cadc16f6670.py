import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_delta_rule_bwd_kernel(q, k, v, beta, h0, dh0, dht, do,
    dq, dk, dv, db, scale, B: 'tl.constexpr', T: 'tl.constexpr', H:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NK: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr', IS_BETA_HEADWISE: 'tl.constexpr',
    USE_DH0: 'tl.constexpr', USE_DHT: 'tl.constexpr', HEAD_FIRST:
    'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    if HEAD_FIRST:
        p_q = q + i_bh * T * K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_k = k + i_bh * T * K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_v = v + i_bh * T * V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_do = do + i_bh * T * V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_dk = dk + (i_v * B * H + i_bh) * T * K + i_k * BK + tl.arange(0, BK
            ) + (T - 1) * K
        p_dv = dv + (i_k * B * H + i_bh) * T * V + i_v * BV + tl.arange(0, BV
            ) + (T - 1) * V
        if IS_BETA_HEADWISE:
            p_beta = beta + i_bh * T * V + i_v * BV + tl.arange(0, BV) + (T - 1
                ) * V
            p_dbeta = db + (i_v * NK * B * H + i_k * B * H + i_bh
                ) * T * V + tl.arange(0, BV) + (T - 1) * V
        else:
            p_beta = beta + i_bh * T + T - 1
            p_dbeta = db + (i_v * B * H + i_bh) * T + T - 1
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
        if IS_BETA_HEADWISE:
            p_beta = beta + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(
                0, BV) + (T - 1) * H * V
            p_dbeta = db + (i_v * NK * B + i_k * B + i_b
                ) * T * H * V + i_h * V + tl.arange(0, BV) + (T - 1) * H * V
        else:
            p_beta = beta + i_b * T * H + (T - 1) * H + i_h
            p_dbeta = db + (i_v * B + i_b) * T * H + (T - 1) * H + i_h
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        d_h += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_do = tl.load(p_do, mask=mask_v, other=0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0)
        else:
            b_beta = tl.load(p_beta)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)
        d_beta = d_v * b_v if IS_BETA_HEADWISE else tl.sum(d_v * b_v)
        d_v = d_v * b_beta
        tl.store(p_dk, d_k, mask=mask_k)
        tl.store(p_dv, d_v, mask=mask_v)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, d_beta, mask=mask_v)
        else:
            tl.store(p_dbeta, d_beta)
        d_h -= b_k[:, None] * d_v[None, :]
        p_q -= K if HEAD_FIRST else H * K
        p_k -= K if HEAD_FIRST else H * K
        p_v -= V if HEAD_FIRST else H * V
        p_do -= V if HEAD_FIRST else H * V
        p_dk -= K if HEAD_FIRST else H * K
        p_dv -= V if HEAD_FIRST else H * V
        p_dbeta -= (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)
        p_beta -= (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)
    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, d_h, mask=mask_k[:, None] & mask_v[None, :])
    tl.debug_barrier()
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if HEAD_FIRST:
        p_q = q + i_bh * T * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_bh * T * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_bh * T * V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_bh * T * V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_bh * T
        p_do = do + i_bh * T * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B * H + i_bh) * T * K + i_k * BK + tl.arange(0, BK)
        p_dk = dk + (i_v * B * H + i_bh) * T * K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k * B * H + i_bh) * T * V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(
                0, BV)
        else:
            p_beta = beta + i_b * T * H + i_h
        p_do = do + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B + i_b
            ) * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_dk = dk + (i_v * B + i_b
            ) * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k * B + i_b
            ) * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
    if USE_INITIAL_STATE:
        mask_h = mask_k[:, None] & mask_v[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0)
    for _ in range(0, T):
        d_k = tl.load(p_dk, mask=mask_k, other=0)
        d_v = tl.load(p_dv, mask=mask_v, other=0)
        d_k -= tl.sum(d_v[None, :] * b_h, axis=1)
        tl.store(p_dk, d_k, mask=mask_k)
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_do = tl.load(p_do, mask=mask_v, other=0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0)
        else:
            b_beta = tl.load(p_beta)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        d_q = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_k)
        p_k += K if HEAD_FIRST else H * K
        p_v += V if HEAD_FIRST else H * V
        p_do += V if HEAD_FIRST else H * V
        p_dq += K if HEAD_FIRST else H * K
        p_dk += K if HEAD_FIRST else H * K
        p_dv += V if HEAD_FIRST else H * V
        p_beta += (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)

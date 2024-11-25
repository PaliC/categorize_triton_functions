import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_bwd_kernel(q, k, v, alpha, beta, ha, dht, dh0, do, dq,
    dk, dv, dalpha, dbeta, dha, h0, s_qk_h, s_vo_h, NK, scale, B, H, T, K:
    'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', USE_DH0:
    'tl.constexpr', USE_DHT: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_ha = ha + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_alpha = alpha + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_beta = beta + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        T - 1) * K
    p_dbeta = dbeta + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(
        0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        T - 1) * V
    p_dha = dha + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV
        ) + (T - 1) * V
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_beta = tl.load(p_beta, mask=mask_bk, other=0)
        b_alpha = tl.load(p_alpha, mask=mask_bk, other=0)
        b_ha = tl.load(p_ha, mask=mask_bv, other=0)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * b_v[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)
        tl.store(p_dk, d_k, mask=mask_bk)
        tl.store(p_dv, d_v, mask=mask_bv)
        b_dha = tl.sum(d_h * b_beta[:, None], axis=0)
        tl.store(p_dha, b_dha, mask=mask_bv)
        b_dbeta = tl.sum(d_h * b_ha[None, :], axis=1)
        tl.store(p_dbeta, b_dbeta, mask=mask_bk)
        d_h += b_dha[None, :] * b_alpha[:, None]
        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V
        p_beta -= K
        p_dbeta -= K
        p_alpha -= K
        p_dha -= V
        p_ha -= V
    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, d_h, mask=mask_bk[:, None] & mask_bv[None, :])
    tl.debug_barrier()
    h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_beta = beta + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_ha = ha + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dha = dha + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_alpha = alpha + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(
        0, BK)
    p_dalpha = dalpha + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(
        0, BK)
    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0)
    for i in range(0, T):
        d_ha = tl.load(p_dha, mask=mask_bv, other=0)
        d_alpha = tl.sum(d_ha[None, :] * h, axis=1)
        tl.store(p_dalpha, d_alpha, mask=mask_bk)
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_beta = tl.load(p_beta, mask=mask_bk, other=0)
        b_ha = tl.load(p_ha, mask=mask_bv, other=0)
        h += b_k[:, None] * b_v[None, :] + b_beta[:, None] * b_ha[None, :]
        _d_q = h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_bk)
        p_k += K
        p_do += V
        p_v += V
        p_dk += K
        p_dalpha += K
        p_dha += V
        p_ha += V
        p_dq += K
        p_beta += K

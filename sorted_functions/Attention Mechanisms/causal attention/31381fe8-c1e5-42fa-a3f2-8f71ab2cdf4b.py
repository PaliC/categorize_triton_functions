import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_bwd_kernel(q, k, v, beta, do, dq, dk, dv, dbeta,
    initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T,
    scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV:
    'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_beta = beta + i_bh * T + T - 1
    p_dbeta = dbeta + (i_bh + i_v * B * H) * T + T - 1
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        T - 1) * DK
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        T - 1) * DV
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0)
        _q = tl.load(p_q, mask=mask_bk, other=0) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _beta = tl.load(p_beta)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :] * _beta, axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)
        d_beta = tl.sum(d_v * _v)
        d_v = d_v * _beta
        tl.store(p_dk, d_k, mask=mask_bk)
        tl.store(p_dv, d_v, mask=mask_bv)
        tl.store(p_dbeta, d_beta)
        d_h -= _k[:, None] * d_v[None, :]
        p_do -= DV
        p_q -= DK
        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV
        p_dbeta -= 1
        p_beta -= 1
    tl.debug_barrier()
    h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_beta = beta + i_bh * T
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV
        ) + DV
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK
        ) + DK
    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0,
            BK)[:, None]) * DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0)
    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _do = tl.load(p_do, mask=mask_bv, other=0)
        _beta = tl.load(p_beta)
        _v *= _beta
        h += _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_bk)
        if i < T - 1:
            d_k = tl.load(p_dk, mask=mask_bk, other=0)
            d_v = tl.load(p_dv, mask=mask_bv, other=0)
            d_k -= tl.sum(d_v[None, :] * h, axis=1)
            tl.store(p_dk, d_k, mask=mask_bk)
        p_k += DK
        p_do += DV
        p_v += DV
        p_dk += DK
        p_dv += DV
        p_dq += DK
        p_beta += 1

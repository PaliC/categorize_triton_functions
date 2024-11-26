import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_gla_bwd_kernel(q, k, v, gk, gv, do, dq, dk, dv,
    initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T,
    scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV:
    'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', REVERSE:
    'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * DK if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) *
            DK if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) *
            DV if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0,
            BK)[:, None]) * DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0)
    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _do = tl.load(p_do, mask=mask_bv, other=0)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0)
            h = h * _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0)
            h = h * _gv[None, :]
        h += _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_bk)
        p_k += -DK if REVERSE else DK
        p_v += -DV if REVERSE else DV
        p_q += -DK if REVERSE else DK
        p_do += -DV if REVERSE else DV
        p_dq += -DK if REVERSE else DK
        if USE_GK:
            p_gk += -DK if REVERSE else DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV
    tl.debug_barrier()
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        not REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        not REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        not REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * DK if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * DV if not REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) *
            DK if not REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) *
            DV if not REVERSE else 0)
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0)
        _q = tl.load(p_q, mask=mask_bk, other=0) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0)
            d_h *= _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0)
            d_h *= _gv[None, :]
        tl.store(p_dk, d_k, mask=mask_bk)
        tl.store(p_dv, d_v, mask=mask_bv)
        p_do += DV if REVERSE else -DV
        p_q += DK if REVERSE else -DK
        p_k += DK if REVERSE else -DK
        p_v += DV if REVERSE else -DV
        p_dk += DK if REVERSE else -DK
        p_dv += DV if REVERSE else -DV
        if USE_GK:
            p_gk += DK if REVERSE else -DK
        if USE_GV:
            p_gv += DV if REVERSE else -DV

import triton
import triton.language as tl
import torch

@triton.jit
def bid_fused_recurrent_gla_fwd_kernel(q, k, v, gk, gv, o, initial_state,
    final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T,
    scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV:
    'tl.constexpr', NK: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    STORE_FINAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr', USE_GK:
    'tl.constexpr', USE_GV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + 0
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + 0
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + 0
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + 0
    inv_p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    inv_p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    inv_p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    inv_p_o = o + (i_bh + i_k * B * H + NK * B * H
        ) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + 0
        inv_p_gk = gk + B * H * s_qk_h + i_bh * s_qk_h + i_k * BK + tl.arange(
            0, BK) + (T - 1) * DK
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) *
            DV if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    h = tl.zeros([BV, BK], dtype=tl.float32)
    inv_h = tl.zeros([BV, BK], dtype=tl.float32)
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0,
            BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0)
    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _q = tl.load(p_q, mask=mask_bk, other=0) * scale
        _inv_k = tl.load(inv_p_k, mask=mask_bk, other=0)
        _inv_v = tl.load(inv_p_v, mask=mask_bv, other=0)
        _inv_q = tl.load(inv_p_q, mask=mask_bk, other=0) * scale
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0)
            h = h * _gk[None, :]
            _inv_gk = tl.load(inv_p_gk, mask=mask_bk, other=0)
            inv_h = inv_h * _inv_gk[None, :]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0)
            h = h * _gv[:, None]
        h += _k[None, :] * _v[:, None]
        inv_h += _inv_k[None, :] * _inv_v[:, None]
        _o = h * _q[None, :]
        _inv_o = inv_h * _inv_q[None, :]
        _o = tl.sum(_o, axis=1)
        _inv_o = tl.sum(_inv_o, axis=1)
        tl.store(p_o, _o, mask=mask_bv)
        tl.store(inv_p_o, _inv_o, mask=mask_bv)
        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV
        inv_p_q += -DK
        inv_p_k += -DK
        inv_p_o += -DV
        inv_p_v += -DV
        if USE_GK:
            p_gk += DK
            inv_p_gk += -DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV
    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + (i_k * BK + tl.arange(0,
            BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h, mask=mask_kv)

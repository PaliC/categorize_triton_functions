import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_gsa_fwd_kernel(q, k, v, gk, gv, o, h0, ht, s_k_h, s_v_h,
    scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE:
    'tl.constexpr', REVERSE: 'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV:
    'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if
        REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if
        REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if
        REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) *
            K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) *
            V if REVERSE else 0)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    mask_h = mask_k[None, :] & mask_v[:, None]
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0)
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0)
            b_h = b_h * tl.exp(b_gk)[None, :]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0)
            b_h = b_h * tl.exp(b_gv)[:, None]
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o, mask=mask_v)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h, mask=mask_h)

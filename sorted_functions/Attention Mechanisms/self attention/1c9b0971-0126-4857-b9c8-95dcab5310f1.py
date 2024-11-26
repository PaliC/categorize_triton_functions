import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(q, k, v, w, u, o, h0, ht, s_k_h, s_v_h,
    scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K:
    'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE:
    'tl.constexpr', REVERSE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if
        REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if
        REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if
        REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if
        REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0)
    b_u = tl.load(p_u, mask=mask_bk, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_w = tl.load(p_w, mask=mask_bk, other=0)
        b_w = tl.exp(b_w)
        b_kv = b_k[None, :] * b_v[:, None]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_o, b_o, mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h, mask=mask_kv)

import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dkv(q, k, v, w, u, do, dk, dk_aux, dv,
    dh0, s_k_h, s_v_h, scale, B: 'tl.constexpr', H: 'tl.constexpr', T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    REVERSE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if 
        not REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if 
        not REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if
        not REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if 
        not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    p_dk_aux = dk_aux + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(
        0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if not REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if 
        not REVERSE else 0)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    b_u = tl.load(p_u, mask=mask_bk, other=0)
    for _ in range(T - 1, -1, -1):
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_w = tl.load(p_w, mask=mask_bk, other=0)
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_dkv = b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        tl.store(p_dk_aux, b_dk, mask=mask_bk)
        b_dk += tl.sum(b_dkv * b_u[:, None] * b_v[None, :], axis=1)
        b_dv = tl.sum((b_dh + b_dkv * b_u[:, None]) * b_k[:, None], axis=0)
        tl.store(p_dk, b_dk, mask=mask_bk)
        tl.store(p_dv, b_dv, mask=mask_bv)
        b_dh *= tl.exp(b_w)[:, None]
        b_dh += b_dkv
        p_q += K if REVERSE else -K
        p_k += K if REVERSE else -K
        p_v += V if REVERSE else -V
        p_w += K if REVERSE else -K
        p_do += V if REVERSE else -V
        p_dk += K if REVERSE else -K
        p_dk_aux += K if REVERSE else -K
        p_dv += V if REVERSE else -V
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]
            ) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh, mask=mask_kv)

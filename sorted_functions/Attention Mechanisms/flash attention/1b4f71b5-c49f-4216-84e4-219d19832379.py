import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_fwd_kernel(q, k, v, alpha, beta, o, ha, h0, ht, s_k_h,
    s_v_h, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV)
    p_alpha = alpha + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_beta = beta + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV)
    p_ha = ha + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_alpha = tl.load(p_alpha, mask=mask_bk, other=0)
        b_beta = tl.load(p_beta, mask=mask_bk, other=0)
        tmp = tl.sum(h * b_alpha[None, :], axis=1)
        h += tmp[:, None] * b_beta[None, :] + b_k[None, :] * b_v[:, None]
        _o = h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o, mask=mask_bv)
        tl.store(p_ha, tmp, mask=mask_bv)
        p_q += K
        p_k += K
        p_o += V
        p_v += V
        p_ha += V
        p_alpha += K
        p_beta += K
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h, mask=mask_kv)

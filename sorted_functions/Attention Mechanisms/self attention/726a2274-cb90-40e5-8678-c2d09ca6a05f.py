import triton
import triton.language as tl
import torch

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None})
@triton.jit
def fused_recurrent_retention_fwd_kernel(q, k, v, o, h0, ht, scale, B:
    'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr',
    V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr',
    HEAD_FIRST: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
    if HEAD_FIRST:
        p_q = q + i_bh * T * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_bh * T * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_bh * T * V + i_v * BV + tl.arange(0, BV)
        p_o = o + (i_k * B * H + i_bh) * T * V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_o = o + (i_k * B + i_b) * T * H * V + i_h * V + i_v * BV + tl.arange(
            0, BV)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0)
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_h = b_b * b_h + b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o, mask=mask_v)
        p_q += K if HEAD_FIRST else H * K
        p_k += K if HEAD_FIRST else H * K
        p_v += V if HEAD_FIRST else H * V
        p_o += V if HEAD_FIRST else H * V
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]
            ) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h, mask=mask_h)

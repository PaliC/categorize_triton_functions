import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_gsa_inference_kernel(q, k, v, s, g, o, hk0, hv0, hkt,
    hvt, scale, K: 'tl.constexpr', V: 'tl.constexpr', M: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr', NG: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG
    b_s = tl.load(s + i_bg * M + tl.arange(0, M))
    b_g = tl.load(g + i_bg * M + tl.arange(0, M))
    b_g = tl.exp(b_g)
    b_ok = tl.zeros([M], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_hk0 = hk0 + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None
            ]
        mask_k = o_k < K
        mask_hk = (tl.arange(0, M) < M)[:, None] & mask_k[None, :]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.0)
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.0) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.0)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)
        if i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[
                :, None]
            tl.store(p_hkt, b_hk, mask=mask_hk)
    b_qv = tl.softmax(b_ok)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_hv0 = hv0 + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None
            ]
        mask_v = o_v < V
        mask_hv = mask_v[:, None] & (tl.arange(0, M) < M)[None, :]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0)
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)
        tl.store(o + i_bh * V + o_v, b_ov, mask=mask_v)
        if i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[
                :, None]
            tl.store(p_hvt, b_hv, mask=mask_hv)

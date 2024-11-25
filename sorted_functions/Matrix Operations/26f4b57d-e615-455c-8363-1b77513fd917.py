import triton
import triton.language as tl
import torch

@triton.jit
def prepare_qg_kg(q, k, g, qg, kg, s_qk_h, scale, K: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    mask = i_k * BK + tl.arange(0, BK) < K
    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * K + i_k *
        BK + tl.arange(0, BK))
    for i in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0)
        b_q *= tl.exp(_g) * scale
        b_k *= tl.exp(last_decay - _g)
        tl.store(p_kg, b_k, mask=mask)
        tl.store(p_qg, b_q, mask=mask)
        p_q += K
        p_g += K
        p_k += K
        p_kg += K
        p_qg += K

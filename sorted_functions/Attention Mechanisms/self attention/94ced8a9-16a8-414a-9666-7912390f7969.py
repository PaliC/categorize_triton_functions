import triton
import triton.language as tl
import torch

@triton.jit
def prepare_qg_kg(q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: 'tl.constexpr', BK: 'tl.constexpr', DK: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    mask = i_k * BK + tl.arange(0, BK) < DK
    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK + i_k *
        BK + tl.arange(0, BK))
    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(last_decay - _g)
        tl.store(p_kg, _k, mask=mask)
        tl.store(p_qg, _q, mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK

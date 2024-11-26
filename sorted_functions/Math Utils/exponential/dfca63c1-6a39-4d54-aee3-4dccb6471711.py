import triton
import triton.language as tl
import torch

@triton.jit
def fwd_decay_cumsum(g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale, BT:
    'tl.constexpr', BK: 'tl.constexpr', DK: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_go = g_o + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = i_k * BK + tl.arange(0, BK) < DK
    for i in range(BT):
        _g = tl.load(p_g, mask=mask, other=0)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay, mask=mask)
        p_g += DK
        p_go += DK

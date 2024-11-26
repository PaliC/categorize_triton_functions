import triton
import triton.language as tl
import torch

@triton.jit
def bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g,
    dg, s_k_h, BT: 'tl.constexpr', BK: 'tl.constexpr', K: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1
        ) * K
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1
        ) * K
    p_g = g + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1
        ) * K
    p_dg = dg + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1
        ) * K
    p_dq_inner = dq_inner + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        i_c * BT + BT - 1) * K
    p_dk_inner = dk_inner + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        i_c * BT + BT - 1) * K
    p_dq_inter = dq_inter + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        i_c * BT + BT - 1) * K
    p_dk_inter = dk_inter + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        i_c * BT + BT - 1) * K
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = i_k * BK + tl.arange(0, BK) < K
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(BT - 1, -1, -1):
        _g = tl.load(p_g, mask=mask, other=0)
        if j == BT - 1:
            last_g = _g
        b_dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        b_dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        b_dq2 *= tl.exp(_g)
        b_dq = b_dq1 + b_dq2
        tl.store(p_dq_inter, b_dq, mask=mask)
        b_dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        b_dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        b_dk2 *= tl.exp(last_g - _g)
        b_dk = b_dk1 + b_dk2
        tl.store(p_dk_inter, b_dk, mask=mask)
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        b_dg = b_dq * b_q - b_dk * b_k
        cum_grad_dg += b_dg
        tl.store(p_dg, cum_grad_dg, mask=mask)
        p_g -= K
        p_k -= K
        p_q -= K
        p_dq_inner -= K
        p_dk_inner -= K
        p_dq_inter -= K
        p_dk_inter -= K
        p_dg -= K

import triton
import triton.language as tl
import torch

@triton.jit
def fwd_inner_chunk(q, k, g, A, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale, BT:
    'tl.constexpr', BK: 'tl.constexpr', K: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    mask = i_k * BK + tl.arange(0, BK) < K
    o_i = tl.arange(0, BT)
    p_q = q + i_bh * s_qk_h + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_gq = g + i_bh * s_qk_h + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_A = A + (i_bh + i_k * B * H) * (tl.cdiv(T, BT) * BT * BT
        ) + i_t * BT * BT + tl.arange(0, BT)
    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0) * scale
        gq = tl.load(p_gq, mask=mask, other=0)
        s = _q[None, :] * b_k * tl.exp(gq[None, :] - b_g)
        score = tl.sum(s, axis=1)
        score = tl.where(o_i <= i, score, 0)
        tl.store(p_A, score)
        p_q += K
        p_gq += K
        p_A += BT

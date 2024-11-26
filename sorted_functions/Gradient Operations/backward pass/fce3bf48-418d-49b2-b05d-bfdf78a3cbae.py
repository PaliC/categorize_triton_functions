import triton
import triton.language as tl
import torch

@triton.jit
def bwd_inner_chunk(q, k, g, dA, dq, dk, s_qk_h, s_qk_t, s_qk_d, B, H, T,
    scale, BT: 'tl.constexpr', BK: 'tl.constexpr', DK: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    mask = i_k * BK + tl.arange(0, BK) < DK
    o_i = tl.arange(0, BT)
    p_q = q + i_bh * s_qk_h + i_k * BK + i_t * BT * DK + tl.arange(0, BK)
    p_dq = dq + i_bh * s_qk_h + i_k * BK + i_t * BT * DK + tl.arange(0, BK)
    p_gq = g + i_bh * s_qk_h + i_k * BK + i_t * BT * DK + tl.arange(0, BK)
    p_dA = dA + i_bh * (tl.cdiv(T, BT) * BT * BT) + i_t * BT * BT + tl.arange(
        0, BT)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0)
        gq = tl.load(p_gq, mask=mask, other=0)
        score = tl.math.exp2(gq[None, :] - b_g)
        score = tl.where(o_i[:, None] <= i, score, 0)
        _dA = tl.load(p_dA)
        _dA = tl.where(o_i <= i, _dA, 0)
        b_dk += _dA[:, None] * score * _q[None, :]
        b_dq = tl.sum(_dA[:, None] * score * b_k, axis=0)
        tl.store(p_dq, b_dq, mask=mask)
        p_q += DK
        p_dq += DK
        p_gq += DK
        p_dA += BT
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))

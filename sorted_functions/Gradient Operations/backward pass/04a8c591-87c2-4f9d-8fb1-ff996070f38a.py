import triton
import triton.language as tl
import torch

@triton.jit
def bwd_prepare_wy_repr(A, cumsum, cumdecay, d_cumsum, d_cumdecay, dA, NT,
    DK, BT: 'tl.constexpr', BK: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_dcumsum = d_cumsum + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0,
        BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_dcumdecay = d_cumdecay + i_bh * NT * BT * DK + (i_t * BT + tl.arange(
        0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_cumsum = cumsum + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[
        :, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_cumdecay = cumdecay + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0,
        BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    o = tl.load(p_cumsum)
    o2 = tl.load(p_cumdecay)
    do = tl.load(p_dcumsum)
    do2 = tl.load(p_dcumdecay)
    p_A = A + i_bh * NT * BT * BT + i_t * BT * BT + tl.arange(0, BT) + (BT - 1
        ) * BT
    p_dA = dA + i_bh * NT * BT * BT + i_t * BT * BT + tl.arange(0, BT) + (BT -
        1) * BT
    for i in range(BT - 1, -1, -1):
        attn = tl.load(p_A)
        mask = tl.arange(0, BT) < i
        attn = tl.where(mask, attn, 0)
        mask2 = tl.arange(0, BT) == i
        do_ = tl.sum(tl.where(mask2[:, None], do, 0), axis=0)
        do2_ = tl.sum(tl.where(mask2[:, None], do2, 0), axis=0)
        dA_ = tl.where(mask[:, None], o, 0) * do_[None, :] + tl.where(mask[
            :, None], o2, 0) * do2_[None, :]
        dA_ = tl.sum(dA_, axis=1)
        tl.store(p_dA, -dA_)
        do = do - attn[:, None] * do_[None, :]
        do2 = do2 - attn[:, None] * do2_[None, :]
        p_A -= BT
        p_dA -= BT
    p_dcumsum = d_cumsum + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0,
        BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_dcumdecay = d_cumdecay + i_bh * NT * BT * DK + (i_t * BT + tl.arange(
        0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_dcumsum, do)
    tl.store(p_dcumdecay, do2)

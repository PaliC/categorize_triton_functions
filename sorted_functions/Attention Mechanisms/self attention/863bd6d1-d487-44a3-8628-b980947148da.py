import triton
import triton.language as tl
import torch

@triton.jit
def fwd_prepare_wy_repr(A, x, k, cumsum, cumdecay, NT, DK, BT:
    'tl.constexpr', BK: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_x = x + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]
        ) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_k = k + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]
        ) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    S = tl.load(p_x)
    p_A = A + i_bh * NT * BT * BT + i_t * BT * BT + tl.arange(0, BT)
    S_cumdecay = tl.load(p_k)
    for i in range(BT):
        attn = tl.load(p_A)
        mask = tl.arange(0, BT) < i
        attn = tl.where(mask, attn, 0)
        new = tl.sum(attn[:, None] * S, axis=0)
        new_cumdecay = tl.sum(attn[:, None] * S_cumdecay, axis=0)
        mask = tl.arange(0, BT) == i
        S = tl.where(mask[:, None], S - new[None, :], S)
        S_cumdecay = tl.where(mask[:, None], S_cumdecay - new_cumdecay[None,
            :], S_cumdecay)
        p_A += BT
    p_cumsum = cumsum + i_bh * BT * NT * DK + (i_t * BT + tl.arange(0, BT)[
        :, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_cumsum, S)
    p_cumdecay = cumdecay + i_bh * BT * NT * DK + (i_t * BT + tl.arange(0,
        BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_cumdecay, S_cumdecay)

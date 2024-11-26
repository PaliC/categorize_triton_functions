import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BD': 32}, num_warps=1), triton.
    Config({'BD': 32}, num_warps=2), triton.Config({'BD': 32}, num_warps=4),
    triton.Config({'BD': 32}, num_warps=8), triton.Config({'BD': 64},
    num_warps=1), triton.Config({'BD': 64}, num_warps=2), triton.Config({
    'BD': 64}, num_warps=4), triton.Config({'BD': 64}, num_warps=8), triton
    .Config({'BD': 128}, num_warps=1), triton.Config({'BD': 128}, num_warps
    =2), triton.Config({'BD': 128}, num_warps=4), triton.Config({'BD': 128},
    num_warps=8)], key=['D'])
@triton.jit
def fused_recurrent_hgrn_bwd_kernel(g, o, dx, dg, do, h0, T: 'tl.constexpr',
    D: 'tl.constexpr', BD: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_g = g + (i_b * T + T - 1) * D + o_d
    p_o = o + (i_b * T + T - 2) * D + o_d
    p_dx = dx + (i_b * T + T - 1) * D + o_d
    p_dg = dg + (i_b * T + T - 1) * D + o_d
    p_do = do + (i_b * T + T - 1) * D + o_d
    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(T - 1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0)
        b_do = tl.load(p_do, mask=mask, other=0)
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_b * D + o_d, mask=mask, other=0)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * tl.exp(b_g)
        b_dg = b_dh * b_o
        tl.store(p_dx, b_dx, mask=mask)
        tl.store(p_dg, b_dg, mask=mask)
        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D

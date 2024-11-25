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
def fused_recurrent_hgrn_fwd_kernel(x, g, o, h0, ht, T: 'tl.constexpr', D:
    'tl.constexpr', BD: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    STORE_FINAL_STATE: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_x = x + i_b * T * D + o_d
    p_g = g + i_b * T * D + o_d
    p_o = o + i_b * T * D + o_d
    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_b * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0)
        b_g = tl.load(p_g, mask=mask, other=0)
        b_h = tl.exp(b_g) * b_h + b_x
        tl.store(p_o, b_h, mask=mask)
        p_x += D
        p_g += D
        p_o += D
    if STORE_FINAL_STATE:
        p_ht = ht + i_b * D + o_d
        tl.store(p_ht, b_h, mask=mask)

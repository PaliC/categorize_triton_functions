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
def chunk_hgrn_fwd_kernel_h(x, g, gc, o, h0, T: 'tl.constexpr', D:
    'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr',
    USE_INITIAL_STATE: 'tl.constexpr'):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_x = x + i_b * T * D + i_t * BT * D + o_d
    p_g = g + i_b * T * D + i_t * BT * D + o_d
    p_gc = gc + i_b * T * D + i_t * BT * D + o_d
    p_o = o + i_b * T * D + i_t * BT * D + o_d
    b_h = tl.zeros([BD], dtype=tl.float32)
    b_gc = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(h0 + i_b * D + o_d, mask=mask, other=0)
    for i in range(0, BT):
        mask_t = mask & (i_t * BT + i < T)
        b_x = tl.load(p_x, mask=mask_t, other=0)
        b_g = tl.load(p_g, mask=mask_t, other=0)
        b_h = tl.exp(b_g) * b_h + b_x
        b_gc = b_gc + b_g
        tl.store(p_gc, b_gc, mask=mask_t)
        tl.store(p_o, b_h, mask=mask_t)
        p_x += D
        p_g += D
        p_gc += D
        p_o += D

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
def chunk_hgrn_bwd_kernel_h(g, gc, dx, do, T: 'tl.constexpr', D:
    'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr'):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    BC = min(BT, T - i_t * BT)
    NT = tl.num_programs(1)
    p_g = g + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_gc = gc + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_dx = dx + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_do = do + (i_b * T + i_t * BT + BC - 1) * D + o_d
    if i_t == NT - 1:
        b_gc = tl.zeros([BD], dtype=tl.float32)
    else:
        b_gc = tl.load(g + (i_b * T + i_t * BT + BT) * D + o_d, mask=mask,
            other=0)
    b_dh = tl.zeros([BD], dtype=tl.float32)
    for _ in range(BC - 1, -1, -1):
        tl.store(p_gc, b_gc, mask=mask)
        b_g = tl.load(p_g, mask=mask, other=0)
        b_do = tl.load(p_do, mask=mask, other=0)
        b_gc = b_gc + b_g
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * tl.exp(b_g)
        tl.store(p_dx, b_dx, mask=mask)
        p_g -= D
        p_gc -= D
        p_dx -= D
        p_do -= D

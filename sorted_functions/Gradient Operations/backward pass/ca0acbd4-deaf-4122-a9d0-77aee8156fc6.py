import triton
import triton.language as tl
import torch

@triton.jit
def chunk_hgrn_bwd_kernel_o(g, gc, o, dx, dg, s_b, s_t, s_d, T:
    'tl.constexpr', D: 'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_g = tl.make_block_ptr(g + i_b * s_b, (T, D), (s_t, s_d), (i_t *
            BT, i_d * BD), (BT, BD), (1, 0))
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t *
            BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t *
            BT - 1, i_d * BD), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_b * s_b, (T, D), (s_t, s_d), (i_t *
            BT, i_d * BD), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_b * s_b, (T, D), (s_t, s_d), (i_t *
            BT, i_d * BD), (BT, BD), (1, 0))
        mask_t = mask & ((i_t + 1) * BT < T)
        b_ht = tl.load(dx + i_b * T * D + (i_t + 1) * BT * D + o_d, mask=
            mask_t, other=0)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_gc = tl.load(p_gc, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))
        b_dx = tl.load(p_dx, boundary_check=(0, 1))
        b_dx = b_dx + tl.exp(b_gc) * b_ht[None, :]
        b_dg = b_o * b_dx * tl.exp(b_g)
        tl.store(p_dx, b_dx, boundary_check=(0, 1))
        tl.store(p_dg, b_dg, boundary_check=(0, 1))

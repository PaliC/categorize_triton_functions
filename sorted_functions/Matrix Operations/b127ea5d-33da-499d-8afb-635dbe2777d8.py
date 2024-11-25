import triton
import triton.language as tl
import torch

@triton.jit
def chunk_hgrn_fwd_kernel_o(gc, o, s_b, s_t, s_d, T: 'tl.constexpr', D:
    'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    for i_t in range(1, tl.cdiv(T, BT)):
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t *
            BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t *
            BT, i_d * BD), (BT, BD), (1, 0))
        b_h0 = tl.load(o + i_b * T * D + i_t * BT * D - D + o_d, mask=mask,
            other=0)
        b_gc = tl.load(p_gc, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))
        b_o = b_o + tl.exp(b_gc) * b_h0[None, :]
        tl.store(p_o, b_o, boundary_check=(0, 1))

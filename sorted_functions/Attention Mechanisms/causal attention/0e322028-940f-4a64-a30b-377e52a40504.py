import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_fwd_kernel_intra_K(v, g, o, A, T: 'tl.constexpr', V:
    'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV:
    'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    if i_t * BT + i_i * BC > T:
        return
    p_g = tl.make_block_ptr(g + i_bg * T * V, (T, V), (V, 1), (i_t * BT + 
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T * V + min(i_t * BT +
        i_i * BC, T) * V + o_v, BV), BV)
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bg * T * V, (T, V), (V, 1), (i_t * BT +
            i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_gv = tl.make_block_ptr(g + i_bg * T * V, (T, V), (V, 1), (i_t *
            BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = b_v * tl.exp(b_gn[None, :] - b_gv)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, b_vg)
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o *= tl.exp(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * BT + i_i * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * T * V + (i_t * BT +
            i_i * BC + j) * V + o_v, BV), BV)
        p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * T * V + (i_t *
            BT + i_i * BC + j) * V + o_v, BV), BV)
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        b_v = tl.load(p_v, mask=m_v, other=0)
        b_gv = tl.load(p_gv, mask=m_v, other=0)
        b_vg = b_v[None, :] * tl.exp(b_g - b_gv[None, :])
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.0)
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT + 
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))
    tl.store(p_o, b_o, boundary_check=(0, 1))

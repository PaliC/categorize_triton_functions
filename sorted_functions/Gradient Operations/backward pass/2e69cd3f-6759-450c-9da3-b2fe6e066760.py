import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_bwd_kernel_intra_KV(v, g, o, A, do, dv, dg, T: 'tl.constexpr',
    V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV:
    'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr', OVERWRITE_DG:
    'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    if i_t * BT + i_i * BC > T:
        return
    p_gv = tl.make_block_ptr(g + i_bg * T * V, (T, V), (V, 1), (i_t * BT + 
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T * V + min(i_t * BT +
        i_i * BC + BC, T) * V - V + o_v, BV), BV)
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_g = tl.make_block_ptr(g + i_bg * T * V, (T, V), (V, 1), (i_t * BT +
            i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (i_i *
            BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t *
            BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_g - b_gn[None, :])
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do)
    b_dv *= tl.exp(b_gn[None, :] - b_gv)
    o_i = tl.arange(0, BC)
    o_c = i_i * BC + tl.arange(0, BC)
    p_g = tl.max_contiguous(tl.multiple_of(g + i_bg * T * V + (i_t * BT + 
        i_i * BC) * V + o_v, BV), BV)
    p_A = tl.max_contiguous(tl.multiple_of(A + i_bh * T * BT + (i_t * BT + 
        i_i * BC) * BT + o_c, BC), BC)
    p_do = tl.max_contiguous(tl.multiple_of(do + i_bh * T * V + (i_t * BT +
        i_i * BC) * V + o_v, BV), BV)
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.load(p_A)
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_g[None, :] - b_gv) * b_A[:, None] *
            b_do[None, :], 0.0)
        p_g += V
        p_A += BT
        p_do += V
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT + 
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bg * T * V, (T, V), (V, 1), (i_t * BT + 
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t * BT +
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * T * V, (T, V), (V, 1), (i_t * BT +
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_bh * T * V, (T, V), (V, 1), (i_t * BT +
        i_i * BC, i_v * BV), (BC, BV), (1, 0))
    b_o = tl.load(p_o, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1))
    b_dg = b_o * b_do - b_v * b_dv
    if not OVERWRITE_DG:
        b_dg = b_dg + tl.load(p_dg, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))

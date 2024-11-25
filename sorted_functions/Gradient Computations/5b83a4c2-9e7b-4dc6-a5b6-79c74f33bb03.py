import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_bwd_kernel_intra_K(v, g, do, dA, s_v_h, s_v_t, scale, T:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr'
    ):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC
        ) % NC
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    p_dA = tl.make_block_ptr(dA + (i_bh + i_v * n_bh) * T * BT, (T, BT), (
        BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (V, T), (1, s_v_t), (i_v *
            BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gv = tl.make_block_ptr(g + i_bg * s_v_h, (V, T), (1, s_v_t), (i_v *
            BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t *
            BT + i_i * BC) * V + o_v, BV), BV)
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t *
            BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (
            i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        b_gn = tl.load(p_gn, mask=m_v, other=0.0)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_g - b_gn[None, :]) * scale
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = b_v * tl.exp(b_gn[:, None] - b_gv)
        b_dA = tl.dot(b_do, b_vg)
    elif i_i == i_j:
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t *
            BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (
            i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * s_v_h + (i_t * BT +
            i_j * BC) * V + o_v, BV), BV)
        p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t *
            BT + i_j * BC) * V + o_v, BV), BV)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        m_v = o_v < V
        o_i = tl.arange(0, BC)
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, BC):
            b_v = tl.load(p_v, mask=m_v, other=0)
            b_gv = tl.load(p_gv, mask=m_v, other=0)
            b_dAj = tl.sum(b_do * b_v[None, :] * tl.exp(b_g - b_gv[None, :]), 1
                )
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)
            p_v += V
            p_gv += V
        b_dA = tl.where(m_dA, b_dA, 0.0)
    tl.store(p_dA, b_dA, boundary_check=(0, 1))

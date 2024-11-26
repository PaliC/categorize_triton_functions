import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_intra_K(v, g, o, A, s_v_h, s_v_t, s_v_d, T:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t *
        BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
    b_gn = tl.load(p_gn, boundary_check=(0,))
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_gv = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = b_v * tl.exp(b_gn[None, :] - b_gv)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, b_vg, allow_tf32=False)
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o *= tl.exp(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * BT + i_i * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    for j in range(0, BC):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (1,), ((i_t *
            BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        p_gv = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (1,), ((i_t *
            BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        b_v = tl.load(p_v, boundary_check=(0,))
        b_gv = tl.load(p_gv, boundary_check=(0,))
        b_vg = b_v[None, :] * tl.exp(b_g - b_gv[None, :])
        m_i = o_i[:, None] >= j
        b_o += tl.where(m_i, b_A[:, None] * b_vg, 0.0)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))

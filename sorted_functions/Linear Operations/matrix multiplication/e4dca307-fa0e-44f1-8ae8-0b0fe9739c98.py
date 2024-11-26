import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_intra_K(v, z, o, A, s_v_h, s_v_t, s_v_d, T:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t *
        BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, tl.exp(b_v - b_zn[None, :]), allow_tf32=False)
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_o *= tl.exp(b_zn[None, :] - b_z)
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * BT + i_i * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    for j in range(0, BC):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (1,), ((i_t *
            BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        b_v = tl.load(p_v, boundary_check=(0,))
        m_i = o_i[:, None] >= j
        b_o += tl.where(m_i, b_A[:, None] * tl.exp(b_v[None, :] - b_z), 0)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))

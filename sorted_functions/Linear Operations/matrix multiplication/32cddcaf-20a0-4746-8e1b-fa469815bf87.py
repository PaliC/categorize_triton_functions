import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_intra_K(v, z, do, dA, s_v_h, s_v_t, s_v_d, scale,
    T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC
        ) % NC
    n_bh = tl.num_programs(2)
    if i_i > i_j:
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (
            i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((
            i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_dA = tl.make_block_ptr(dA + (i_bh + i_v * n_bh) * T * BT, (T, BT),
            (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        b_zn = tl.load(p_zn, boundary_check=(0,))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_zn[None, :] - b_z) * scale
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.exp(b_v - b_zn[:, None])
        b_dA = tl.dot(b_do, b_v, allow_tf32=False)
        tl.store(p_dA, b_dA, boundary_check=(0, 1))
    elif i_i == i_j:
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t *
            BT + i_j * BC) * V + i_v * BV,), (BV,), (0,))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_v * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.
            arange(0, BC)) * BT + i_j * BC
        m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
        for j in range(0, BC):
            b_v = tl.load(p_v, boundary_check=(0,))
            b_dA = tl.sum(b_do * tl.exp(b_v[None, :] - b_z), 1)
            b_dA = tl.where(o_i >= j, b_dA, 0)
            tl.store(dA + o_A + j, b_dA, mask=m_A)
            p_v = tl.advance(p_v, (V,))

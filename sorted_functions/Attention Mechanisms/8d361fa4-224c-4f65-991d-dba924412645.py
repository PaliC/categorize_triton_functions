import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_intra_KV(v, z, A, do, dv, s_v_h, s_v_t, s_v_d, T:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t *
        BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t *
        BT + i_i * BC + BC - 1) * V + i_v * BV,), (BV,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
            i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (i_i *
            BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_zn[None, :] - b_z)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do, allow_tf32=False)
    b_dv *= tl.exp(b_v - b_zn[None, :])
    o_i = tl.arange(0, BC)
    for j in range(0, BC):
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (1,), ((i_t *
            BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T * BT,), (1,), ((i_t *
            BT + i_i * BC + j) * BT + i_i * BC,), (BC,), (0,))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T * V,), (1,), ((i_t *
            BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        b_A = tl.load(p_A, boundary_check=(0,))
        b_z = tl.load(p_z, boundary_check=(0,))
        b_do = tl.load(p_do, boundary_check=(0,))
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_v - b_z[None, :]) * b_A[:, None] *
            b_do[None, :], 0.0)
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (
        i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
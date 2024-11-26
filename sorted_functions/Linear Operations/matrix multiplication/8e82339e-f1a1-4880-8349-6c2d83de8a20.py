import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_rcum_intra(s, z, ss, doo, s_s_h, s_s_t, s_s_d, T:
    'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', BS: 'tl.constexpr', NC: 'tl.constexpr'):
    i_s, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    o_i = tl.arange(0, BC)
    m_o = tl.full([BC, BC], 1.0, dtype=tl.float32)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (s_s_d,), ((i_t *
        BT + i_i * BC + BC - 1) * S + i_s * BS,), (BS,), (0,))
    p_doo = tl.make_block_ptr(doo + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
        i_t * BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_doo = tl.zeros([BC, BS], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T, S), (s_s_t, s_s_d),
            (i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_ss = tl.load(p_ss, boundary_check=(0, 1))
        b_doo += b_ss * tl.exp(b_zn[None, :] - b_z)
    b_doo = tl.exp(b_s - b_zn[None, :]) * tl.dot(m_o, b_doo, allow_tf32=False)
    for j in range(0, BC):
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (1,), ((i_t *
            BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T * S,), (1,), ((i_t *
            BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        b_z = tl.load(p_z, boundary_check=(0,))
        b_ss = tl.load(p_ss, boundary_check=(0,))
        m_i = o_i[:, None] <= j
        b_doo += tl.where(m_i, tl.exp(b_s - b_z[None, :]) * b_ss[None, :], 0.0)
    b_doo += tl.load(p_doo, boundary_check=(0, 1))
    tl.store(p_doo, b_doo, boundary_check=(0, 1))

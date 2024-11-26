import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_rcum(s, c, z, s_s_h, s_s_t, s_s_d, T:
    'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, i_s * BS), (BT, BS), (1, 0))
    p_c = tl.make_block_ptr(c + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, i_s * BS), (BT, BS), (1, 0))
    p_z = tl.make_block_ptr(z + i_bh * NT * S, (NT * S,), (s_s_d,), (i_t *
        S + i_s * BS,), (BS,), (0,))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_c = tl.dot(m_s, b_s)
    b_z = tl.sum(b_s, 0)
    tl.store(p_c, b_c, boundary_check=(0, 1))
    tl.store(p_z, b_z, boundary_check=(0,))

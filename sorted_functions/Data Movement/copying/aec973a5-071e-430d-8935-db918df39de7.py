import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_cum(s, o, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr',
    S: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))

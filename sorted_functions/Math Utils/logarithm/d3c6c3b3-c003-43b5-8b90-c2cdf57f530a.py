import triton
import triton.language as tl
import torch

@triton.jit
def logcumsumexp_fwd_kernel(s, z, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S:
    'tl.constexpr', BT: 'tl.constexpr', NT: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_mp = tl.full([S], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([S], dtype=tl.float32)
    for i_t in range(NT):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT, 0), (BT, S), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT, 0), (BT, S), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_mc = tl.max(b_s, 0)
        if i_t > 0:
            b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * tl.exp(b_mp - b_mc)
        b_s = tl.exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        b_z = tl.log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
        tl.store(p_z, b_z, boundary_check=(0, 1))

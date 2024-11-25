import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_rcum_inter(s, z, ss, doo, s_s_h, s_s_t, s_s_d, T:
    'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    b_sp = tl.zeros([BS], dtype=tl.float32)
    b_zp = tl.full([BS], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_zc = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (s_s_d,), (i_t *
            BT * S + i_m * BS,), (BS,), (0,))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T, S), (s_s_t, s_s_d),
            (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_doo = tl.make_block_ptr(doo + i_bh * s_s_h, (T, S), (s_s_t, s_s_d
            ), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        b_zc = tl.load(p_zc, boundary_check=(0,))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_ss = tl.load(p_ss, boundary_check=(0, 1))
        b_doo = tl.exp(b_s - b_zp[None, :]) * b_sp[None, :]
        tl.store(p_doo, b_doo, boundary_check=(0, 1))
        b_sp = b_sp * tl.exp(b_zc - b_zp) + tl.sum(b_ss * tl.exp(b_zc[None,
            :] - b_z), 0)
        b_zp = b_zc

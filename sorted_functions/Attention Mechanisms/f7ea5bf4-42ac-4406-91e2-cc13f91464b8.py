import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_dh(q, z, do, dh, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t,
    s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K: 'tl.constexpr',
    V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NT: 'tl.constexpr', NORMK: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_zp = tl.full([BK if NORMK else BV], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        i_p = tl.maximum(i_t * BT - 1, 0)
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (
            i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (
            s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        if NORMK:
            p_z = tl.make_block_ptr(z + i_bh * s_k_h, (K, T), (s_k_d, s_k_t
                ), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_zc = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,),
                (i_p * K + i_k * BK,), (BK,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zc - b_zp), b_zc
            b_z = tl.load(p_z, boundary_check=(0, 1))
            b_q = b_q * tl.exp(b_zc[:, None] - b_z)
            b_dh = b_dh * b_r[:, None]
        else:
            p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d
                ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,),
                (i_p * V + i_v * BV,), (BV,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zc - b_zp), b_zc
            b_z = tl.load(p_z, boundary_check=(0,))
            b_do = b_do * tl.exp(b_zc[None, :] - b_z)
            b_dh = b_dh * b_r[None, :]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)

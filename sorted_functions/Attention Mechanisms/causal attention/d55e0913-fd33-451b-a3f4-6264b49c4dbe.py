import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_fwd_kernel_cum(s, r, c, p, s_sk_h, s_sk_t, s_sk_m, T, BT:
    'tl.constexpr', BM: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'
    ):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (
        0, i_m * BM), (BT, BM), (1, 0))
    p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,),
        (i_m * BM,), (BM,), (0,))
    p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (
        0, i_m * BM), (BT, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (
        0, i_m * BM), (BT, BM), (1, 0))
    b_mp = tl.zeros([BM], dtype=tl.float32)
    b_zp = tl.zeros([BM], dtype=tl.float32)
    for i in range(NT):
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_m = tl.max(b_s, 0)
        if i == 0:
            b_r = tl.exp(-b_m)
        else:
            b_m = tl.maximum(b_mp, b_m)
            b_r = tl.exp(b_mp - b_m)
        b_c = tl.exp(b_s - b_m[None, :])
        b_z = tl.cumsum(b_c, 0) + (b_zp * b_r)[None, :]
        b_p = tl.exp(-tl.log(b_z))
        b_mp = b_m
        b_zp = tl.max(b_z, 0)
        tl.store(p_r, b_r, boundary_check=(0,))
        tl.store(p_c, b_c, boundary_check=(0, 1))
        tl.store(p_p, b_p, boundary_check=(0, 1))
        p_s = tl.advance(p_s, (BT, 0))
        p_r = tl.advance(p_r, (DM,))
        p_c = tl.advance(p_c, (BT, 0))
        p_p = tl.advance(p_p, (BT, 0))

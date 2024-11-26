import triton
import triton.language as tl
import torch

@triton.jit
def softmax_bwd_kernel(p, dp, ds, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S:
    'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, 0), (BT, S), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
        i_t * BT, 0), (BT, S), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
        i_t * BT, 0), (BT, S), (1, 0))
    b_p = tl.load(p_p, boundary_check=(0, 1))
    b_dp = tl.load(p_dp, boundary_check=(0, 1))
    b_pp = tl.sum(b_p * b_dp, 1)
    b_ds = b_p * b_dp - b_p * b_pp[:, None]
    tl.store(p_ds, b_ds, boundary_check=(0, 1))

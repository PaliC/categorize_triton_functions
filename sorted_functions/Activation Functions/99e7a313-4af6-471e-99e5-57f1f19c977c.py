import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=2), triton.Config({},
    num_warps=4), triton.Config({}, num_warps=8)], key=['S'])
@triton.jit
def softmax_fwd_kernel(s, p, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S:
    'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, 0), (BT, S), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, 0), (BT, S), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_m = tl.max(b_s, 1)
    b_s = tl.exp(b_s - b_m[:, None])
    b_z = tl.sum(b_s, 1)
    b_p = tl.where(b_s != 0, b_s / b_z[:, None], 0.0)
    tl.store(p_p, b_p, boundary_check=(0, 1))

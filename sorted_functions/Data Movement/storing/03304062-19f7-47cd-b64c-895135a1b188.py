import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BC'])
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(A, A2, B: 'tl.constexpr',
    T: 'tl.constexpr', H: 'tl.constexpr', BT: 'tl.constexpr', BC:
    'tl.constexpr', NK: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if i_t * BT + i_c * BC >= T:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        if HEAD_FIRST:
            p_A = tl.make_block_ptr(A + (i_k * B * H + i_bh) * T * BC, (T,
                BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        else:
            p_A = tl.make_block_ptr(A + (i_k * B + i_b) * T * H * BC + i_h *
                BC, (T, BC), (H * BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC
                ), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    if HEAD_FIRST:
        p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    else:
        p_A2 = tl.make_block_ptr(A2 + i_b * T * H * BT + i_h * BT, (T, BT),
            (H * BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A, boundary_check=(0, 1))
